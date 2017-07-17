## Dynamic on-the-fly masking

### Usage

0) You can use already existing masking tracks by passing them with the -i option to the masking server for initialization.

1) Launch the DMserver process with the databse, expected coverage and other optional arguments. The server process then initializes and waits for incoming messages on the default or given hostname and port.

2) Launch the daligner jobs with the -D option. Thereby ensuring daligner notifies the masking server that new las files are available and that daligner fetches masking tracks from the msking server.

3) daligner only reports the path of the finished las file to the masking server. So a shared underlying filesystem is a requirement.

4) When a new las file is available, the server processes it, updates the reads' coverage statistics and derives a new masking track from them. Regions of reads receiving an excess of coverage are masked.

5) You can interact with the server process (e.g. to initiate a shutdown) using the DMctl command line tool.

6) Due to the long-running nature of the server process optional writing of checkpoint files is supported. Checkpoints receive an alternating suffix of either .0 or .1. This ensures at least one intact file in case the server crashes while writing a checkpoint.

7) To resume from a checkpoint remove the suffix, the server will then resume automatically from it upon startup.

### Results

The dynamic masking server produces two masking tracks, maskr and maskc. If the masking was launched with the -C option, which masks contained reads altogether, the maskc track contains the masks for those. If (a part of) a read was masked due to the excessive number of local alignments, then the interval containing the repetetive sequence will be contained in maskr.

If the overlaps are meant to be used for assembly, I recommend to run TKhomogenize with the maskr track in order to transfer the repeat annotation based on the alignments between the reads.

### Requirements

A 50x human genome needs roughly 20GB of memory.

### Notes

Processing the diagonal of the block vs block matrix first is recommended. Thereby ensuring that the most highly repetitive elements in all reads first are masked first.

The las files are large initially when no repeat regions have been masked.

The amount of threads the server process uses needs to appropriate for the number of concurrent daligner jobs. Otherwise the processing of finished las files will start to lag behing their production.

### Tools involved

    DMserver

Maintains coverage statistics for all reads based on .las files produced by daligner and serves mask track.

    DMctl

Interact with the dynamic masking server.

    daligner

When launched with the -D option, queries the DMserver process for the current blocks' masking tracks.

