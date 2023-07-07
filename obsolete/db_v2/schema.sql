
create table read (
                    id           integer primary key autoincrement,
                    sequence     blob not null,
                    length       integer not null,
                    flags        integer default 0
                  );

create table block (
                    id           integer primary key,
                    rid_from integer not null references read,
                    rid_to   integer not null references read,
                    size         integer not null
                   );

create table track (
                    id           integer primary key autoincrement,
                    name         text not null unique
                   );

create table track_data (
                          rid  integer references read,
                          tid integer references track,
                          pos_from integer,
                          pos_to   integer,
                          data     blob
                        );


