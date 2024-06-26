---
title: Scale and performance in a distributed file system
link: afs
published_date: 2021-11-07
meta_description: Description of Andrew file system (AFS) as an example of a distributed file system
tags: afs, distributed-file-systems, distributed-systems, file-systems, storage-systems
---

["Scale and performance in a distributed file
system"](https://www.cs.cmu.edu/~satya/docdir/howard-tocs-afs-1988.pdf)
by Howard et al. was published in ACM Transactions on Computer Systems in 1988.
The paper describes the Andrew File System (AFS). AFS is a distributed file
system which was developed at Carnegie Mellon University (CMU) to support its
users but went on to become widely used at many other universities and
organizations (and it is still used at CMU :)). AFS was a contemporary of NFS
but focused more on scale, which led to fundamentally different design
choices, e.g., in caching. 

**Basic architecture**: AFS consists of a set of file servers that store the shared
data of the file system -- these servers are referred to as Vice. Multiple
clients can access the shared data by communicating with the Vice servers. Each
client runs a process called Venus that acts as a go-between for the
applications on the client machines and the file servers.

AFS uses _file-level caching_. When a client application opens a file, Venus
reads the entire file from the Vice servers and stores it on the client’s disk,
      i.e., it caches the entire file at the client. All subsequent reads and writes
      to the file become cache-hits and proceed as if the file is present on a local
      file system, without any involvement of Venus. Once the application
      closes the file, Venus checks whether the file was updated by the application and 
      updates it on the Vice server. Venus maintains this file-cache using an LRU
      eviction scheme -- the file used farthest in the past is removed from the cache
      to reclaim cache space. 

      The choice of file-level caching was a deviation from the then-prevalent design
      of block-level caching. For example, [NFS](/nfs) 
    used block-level caching, and so did
    all but one of the distributed file systems surveyed by the authors at the time
    of writing the paper. Distributed file systems with block-level caching provide
    lower-latency access to a file because they don’t have to read an entire file
    before an application can access the file. On the flip side, file-level caching
    provides higher throughput access to the file data for applications that access
    multiple blocks in the file. This is because there is only a single network
    communication at the time of the file open. Reduced communications to the
    server also help with scaling, as the server is less utilized, which was one of
    the main goals of AFS.

    AFS’s design was modified based
    on the experiences with a deployed prototype version of AFS. We will discuss
    the design elements that were limiting the AFS prototype performance and how
    the authors modified the design to improve performance. 

    **Server process architecture**: 
    In the AFS prototype, each Venus client had a corresponding process in the Vice
    server. The process was initialized and terminated when the client initiated
    and terminated a network connection with a Vice server. The
    one-process-per-client design offered strong fault isolation -- a process crash
    would affect only a single client. However, the design caused Vice servers
    to hit resource limits (there are limits to the number of processes that a
            server can support). Additionally, the inter-process context switch overhead
    was a limiting factor for the server performance. 

    To address this, AFS designers eschewed the one-process-per-client design in
    favor of a thread-pool like design. In a thread-pool design there is a fixed
    number of threads on the server. Each thread takes up a work item, processes
    that, and moves on to the next work item. For AFS, a work item corresponds to a
    request from a client. In addition to reducing the context switch overhead,
    using threads also led to simpler shared state management as threads (unlike
            processes) can use shared memory. 

    **Callbacks to reduce network communications**:
    The largest fraction of network communication in the AFS prototype came from
    cache validation requests. Before accessing any cached file on the client,
    Venus would check the modification timestamp (mtime) of the file with the Vice
    server. This was to ensure that client applications do not consume data from a
    stale copy of the file. A high amount of network requests is an impediment to
    scalability, so the AFS designers wanted to reduce the number of cache
    validation requests. 

    To reduce the number of cache validation network requests, AFS introduced the
    concept of _callbacks_. A callback is a contract between the server and the
    client that stipulates that the server would let the client know if a file
    cached at the client has been updated at the server. With callbacks, the client
    does not have to validate its cache before using it and it assumes that the
    cache is valid unless told otherwise by the server. Although callbacks require
    the server to be stateful (i.e., maintain information about what clients have
            callbacks on what files), the authors note that the added complexity of
    stateful servers was worth the performance boost from reduced network traffic. 

    **Logical volumes abstraction**:
    An operational problem with the AFS prototype was its use of stub directories
    to stitch together the data stored at different servers. Consider a path
    ``` /user/aha/images/cat.jpeg ```. 
    Further consider that server A stores the data for
    ``` /user/aha ``` and server B stores the data for ``` /images```. For AFS to export the
    desired path (```/user/aha/images/cat.jpeg```), server A stores a stub directory
    named ```images``` under ```/user/aha```. A stub directory means that it does not actually
    contain the data, but rather it contains the information about where to find
    the relevant data, i.e., server B in this case. Using stub directories, the
    location of data was embedded in the file system tree itself. However, because
    the file system tree was glued together arbitrarily using such stub
    directories, it was difficult to move data around. If a server was running low
    on space, moving part of the data to a new server would require a lot of
    operational effort to create and update stub directories. 

    To improve the operability, AFS introduced the concept of _volumes_. A volume was
    a logical abstraction for a collection of files and directories, and typically
    a volume stored data for a single user. AFS also started using a dedicated location
    database to avoid embedding the volume location information in the file system
    tree. The use of volumes helped in keeping the location database to a manageable
    size (storing the location of each file, instead of each volume, would have
            required more space). 
    Volumes also served as the granularity for other data management
    functionalities like per-user space quotas, backups and read-only replication.

    **Reducing path lookup overhead**:
    The last source of performance overhead, as observed in the AFS prototype, was
    the use of file-paths for server-client communications. For any operation on a
    file, Venus passed the entire filename to the Vice server and it was the
    responsibility of the Vice server to convert the file path to an inode number for
    accessing the file data. Measurements with the prototype deployment showed that
    these path lookups were CPU-intensive, leading to high CPU utilization on
    the Vice servers. 

    In the new AFS design, the name resolution work was delegated to the clients,
    and server-client communications used a _fid_ instead of a file path. A fid is a
    combination of the volume number, vnode number and generation number. Each of
    these three components of fid serve a purpose similar to the file handle in
    [NFS](/nfs). 
    The volume number is used to identify the server hosting the data (using
            the location database), the vnode is used to identify a file within the volume,
    and the generation number is used to enable safe reuse of vnode numbers as
    files are deleted. As an aside, a vnode is an abstraction for an inode -- since
    each file system has its own inode structure, the operating system uses vnode
    as a common data structure to support multiple file systems. 
    The vnode contains a pointer to the inode, which is ultimately used for the file system operations. 

    **End-to-end working example**: The paper provides an example of the end-to-end
    working of an AFS file access. Suppose a user on a Venus client wants to access
    a file with pathname ```/user/aha/afs.txt```. The kernel, upon identifying that the
    file belongs to an AFS, will pass the control to Venus. One of
    Venus’ threads would perform the following steps:

    - For each successive component in the path P (i.e., for ```/```, ```/user```, ```/user/aha/```, and ```/user/aha/afs.txt```), Venus will check that the component is in the cache and has a callback established on it. If not, Venus would contact the Vice server to bring the component in the cache and establish a callback on it. After this step, the entire path and the file are in the cache with established callbacks.  
    - Venus hands over control to the kernel and user’s access to the file continues as if the file was present on a local file system.  
    - Once the user closes the file, Venus takes over once again and writes back the file if it  was updated. 

    Note that Venus uses fid instead of path names to communicate with the Vice
    server. The Vice server is unaware of path names and only understands fid. It
    is the responsibility of the Venus client to convert the path components to
    fid. 

    **Cache-consistency**: AFS’s file-level caching and callbacks offer a
    clean cache-consistency semantics. 
    When the server receives an updated file, it informs all the
    clients that have the corresponding file cached with a callback that the file
    has been updated. This is referred to as the server breaking the callback. The
    responsibility of getting the updated copy of the file is left to the client. 

    If two clients are concurrently updating the same file, AFS offers a _last
    writer wins_ semantic. Suppose client A and client B are both updating the
    same file. Consider that client A closes the file first and flushes its updates
    to the server. The server would send a callback break to client B. Further
    consider that for some reason, client B ignores the callback break and
    continues to update the file. When client B closes the file and flushes it 
    to the server, it would “win” in
    the sense that its updates will overwrite any of the updates from client A.
    Realistically, such scenarios can occur in race conditions wherein both the
    clients close the file near instantaneously and before client B receives the
    callback break from client A’s updates, it sends its update to the server -- in
    such a case the last write from client B would win. 

    **Concluding thoughts**: The AFS paper is a great demonstration of a measurement
    and data-driven research methodology. The authors deployed an initial version,
    collected data from it, and used that to design an improved version, which is
    how systems are really built and operated. The design elements of AFS, e.g.,
    file-level caching, callbacks, thread pools, provide useful insights into
    building scalable distributed (file-)systems. 

