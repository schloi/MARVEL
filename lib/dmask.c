#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <netdb.h>
#include <errno.h>

#include "dmask.h"
#include "dmask_proto.h"

static int hostname_to_ip(const char* hostname, char* ip)
  {
    struct hostent *he;
    struct in_addr **addr_list;

    if ((he = gethostbyname(hostname)) == NULL)
      {
        herror("gethostbyname");
        return 0;
      }

    addr_list = (struct in_addr**) he->h_addr_list;

    if (addr_list[0] != NULL)
      {
        strcpy(ip, inet_ntoa(*addr_list[0]));
        return 1;
      }

    return 0;
  }

DynamicMask* dm_init(const char* host, uint16 port)
  {
    int sockfd;
    struct sockaddr_in dest;
    char ip[16];

    if (!hostname_to_ip(host, ip))
      {
        fprintf(stderr, "failed to resolve hostname\n");
        return NULL;
      }

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
      {
        fprintf(stderr, "error creating socket\n");
        return NULL;
      }

    bzero(&dest, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_port = htons(port);

    if (inet_aton(ip, &(dest.sin_addr)) == 0)
      {
        fprintf(stderr, "failed to parse %s:%d\n", host, port);
        return NULL;
      }

    int retry = 3;
    int connected = 0;
    
    while ( retry > 0 && connected != 1 )
    {
      if (connect(sockfd, (struct sockaddr*) &dest, sizeof(dest)) != 0)
      {
        fprintf(stderr, "could not connect to %s:%d - %s\n", host, port, strerror(errno));
        retry--;
        sleep(2);
      }
      else
      {
        connected = 1;
      }
    }
    
    if (connected == 0)
    {
      return NULL;
    }

    DynamicMask* dm = calloc(1, sizeof(DynamicMask));

    dm->sockfd = sockfd;
    dm->dest = dest;
    dm->send_next = 1;

    return dm;
  }

void dm_send_next(DynamicMask* dm, int run, HITS_DB* blocka, char* namea, HITS_DB* blockb, char* nameb)

  {
    char* dira = strdup(getDir(run, blocka->part));
    char* dirb = strdup(getDir(run, blockb->part));

    int send_next = 1;
    char* path;
    int self = (blocka == blockb);

    path = Catenate(Catenate(dira, "/", namea, ""), ".", nameb, ".las");
    if (access(path, F_OK) != -1)
      send_next = 0;

    if (!self)
      {
        path = Catenate(Catenate(dirb, "/", nameb, ""), ".", namea, ".las");

        if (access(path, F_OK) != -1)
          send_next = 0;

      }

    dm->send_next = send_next;
  }

int dm_send_block_done(DynamicMask* dm, int run, HITS_DB* blocka, char* namea, HITS_DB* blockb, char* nameb)
  {
    if (dm->send_next == 0)
      {
        dm->send_next = 1;
        return 0;
      }

    char* dira = strdup(getDir(run, blocka->part));
    char* dirb = strdup(getDir(run, blockb->part));

    int mmax = 1000;
    int mcur = 0;
    char* msg = malloc(mmax);
    char* path;
    char abspath[PATH_MAX + 1];

    int self = (blocka == blockb);

    mcur += sizeof(DmHeader);
    msg[mcur] = '\0';

    int len;
    path = Catenate(Catenate(dira, "/", namea, ""), ".", nameb, ".las");

    if (realpath(path, abspath) == NULL)
      {
        fprintf(stderr, "realpath failed. path: %s, abspath: %s\n", path, abspath);
        exit(1);
      }

    len = strlen(abspath) + 1;

    if (mcur + len > mmax)
      {
        mmax = (mcur + len) * 1.2 + 1000;
        msg = realloc(msg, mmax);
      }

    memcpy(msg + mcur, abspath, len);
    mcur += len;
    if (!self)
      {
        path = Catenate(Catenate(dirb, "/", nameb, ""), ".", namea, ".las");

        if (realpath(path, abspath) == NULL)
          {
            fprintf(stderr, "realpath failed. path: %s, abspath: %s\n", path, abspath);
            exit(1);
          }

        len = strlen(abspath) + 1;

        if (mcur + len > mmax)
          {
            mmax = (mcur + len) * 1.2 + 1000;
            msg = realloc(msg, mmax);
          }

        memcpy(msg + mcur, abspath, len);
        mcur += len;
      }
    DmHeader* header = (DmHeader*) msg;
    header->version = DM_VERSION;
    header->type = DM_TYPE_LAS_AVAILABLE;
    header->length = mcur;

    // TODO ... loop in case send didn't transfer the whole buffer

    if (send(dm->sockfd, msg, mcur, 0) != mcur)
      {
        fprintf(stderr, "failed to send BLOCK DONE message\n");
        return 0;
      }

    free(dira);
    free(dirb);

    return 1;
  }

void dm_free(DynamicMask* dm)
  {
    if (dm == NULL)
      {
        return;
      }

    close(dm->sockfd);

    free(dm);
  }

static int socket_receive(int sock, uint64 data, void* buffer)
  {
    uint64 pending = data;
    uint64 bcur = 0;

    while (pending)
      {
        int received = recv(sock, buffer + bcur, pending, 0);

        if (received < 1)
          {
            fprintf(stderr, "failed to receive\n");
            break;
          }

        bcur += received;
        pending -= received;
      }

    if (pending != 0)
      {
        fprintf(stderr, "%lld pending bytes\n", pending);
      }

    return (pending == 0);
  }

static void dm_simple_message(DynamicMask* dm, int msg)
  {
    DmHeader header;
    bzero(&header, sizeof(header));

    header.version = DM_VERSION;
    header.type = msg;
    header.length = sizeof(header);

    if (send(dm->sockfd, &header, sizeof(header), 0) != sizeof(header))
      {
        fprintf(stderr, "failed to send message\n");
      }
  }

void dm_shutdown(DynamicMask* dm)
  {
    dm_simple_message(dm, DM_TYPE_SHUTDOWN);
  }
  
void dm_write_track(DynamicMask* dm)
{
    dm_simple_message(dm, DM_TYPE_WRITE_TRACK);
}

void dm_lock(DynamicMask* dm)
  {
    dm_simple_message(dm, DM_TYPE_LOCK);
  }

void dm_unlock(DynamicMask* dm)
  {
    dm_simple_message(dm, DM_TYPE_UNLOCK);
  }

void dm_intervals(DynamicMask* dm)
  {
    dm_simple_message(dm, DM_TYPE_INTERVALS);
  }

int dm_done(DynamicMask* dm, char** files)
  {
    if (dm->send_next == 0)
      {
        dm->send_next = 1;
        return 0;
      }

    int mmax = 1000;
    int mcur = 0;
    char* msg = malloc(mmax);
    char abspath[PATH_MAX + 1];

    mcur += sizeof(DmHeader);
    msg[mcur] = '\0';

    int i, len;
    i = 0;
    while (files[i] != NULL)
      {
        if (realpath(files[i], abspath) == NULL)
          {
            perror("realpath failed");
            exit(1);
          }
        len = strlen(abspath) + 1;

        if (mcur + len > mmax)
          {
            mmax = (mcur + len) * 1.2 + 1000;
            msg = realloc(msg, mmax);
          }

        memcpy(msg + mcur, abspath, len);
        mcur += len;
        i++;
      }

    if (mcur == 0)
      {
        free(msg);
        return 0;
      }

    DmHeader* header = (DmHeader*) msg;
    header->version = DM_VERSION;
    header->type = DM_TYPE_LAS_AVAILABLE;
    header->length = mcur;

    // TODO ... loop in case send didn't transfer the whole buffer

    if (send(dm->sockfd, msg, mcur, 0) != mcur)
      {
        fprintf(stderr, "failed to send BLOCK DONE message\n");
        free(msg);
        return 0;
      }

    free(msg);
    return 1;
  }

HITS_TRACK* dm_load_track(HITS_DB* db, DynamicMask* dm, char* trackName)
  {
    DmHeader header;
    bzero(&header, sizeof(header));

    header.version = DM_VERSION;
    header.type = DM_TYPE_REQUEST_TRACK;
    header.length = sizeof(header);

    header.reserved1 = db->ufirst;
    header.reserved2 = db->nreads;

    // track request

    if (send(dm->sockfd, &header, sizeof(header), 0) != sizeof(header))
      {
        fprintf(stderr, "failed to send BLOCK DONE message\n");
        return NULL;
      }

    // response header

    if (recv(dm->sockfd, &header, sizeof(header), 0) != sizeof(header))
      {
        fprintf(stderr, "failed to receive header\n");
        return NULL;
      }

    // track anno part

    uint64 len_anno = sizeof(track_anno) * (db->nreads + 1);

    track_anno* buf_anno = malloc(len_anno);

    if (!socket_receive(dm->sockfd, len_anno, buf_anno))
      {
        fprintf(stderr, "failed to receive track.anno\n");
        return NULL;
      }

    // track data part

    uint64 len_data = buf_anno[db->nreads];
    track_data* buf_data = malloc(len_data);

    if (!socket_receive(dm->sockfd, len_data, buf_data))
      {
        fprintf(stderr, "failed to receive track.data\n");
        return NULL;
      }

    assert((len_anno + len_data + sizeof(DmHeader)) == header.length);

    HITS_TRACK* track = (HITS_TRACK*) malloc(sizeof(HITS_TRACK));

    track->name = strdup(trackName);
    track->size = sizeof(track_anno);
    track->anno = buf_anno;
    track->data = buf_data;

    track->next = db->tracks;
    db->tracks = track;

    printf("received %llu + %llu bytes\n", len_anno, len_data);

    return track;
  }

