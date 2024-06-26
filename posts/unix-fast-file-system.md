---
title: A fast file system for UNIX
link: unix-fast-file-system
published_date: 2021-03-07
meta_description: Description of the UNIX fast file system
tags: file-systems, storage-systems
---

["A fast file system for UNIX"](https://dsf.berkeley.edu/cs262/FFS.pdf) by
McKusick et al. was published in the journal Transaction on Computer Systems in
1984\. It’s take-away message of designing file systems to be aware of the underlying storage medium (HDDs in the case of the fast file system) laid
the foundations for file systems and has carried over to modern file systems
for HDDs, SSDs, and new emerging storage technologies. We will discuss the
disk-aware design of the fast file system (FFS) as described in a paper and in
the (awesome!) [OSTEP
book](https://pages.cs.wisc.edu/~remzi/OSTEP/file-ffs.pdf).

We will begin by discussing some of the basics of file systems and the design
of the file system used in UNIX before FFS, which we will refer to as the old
file system (OFS). Following that, we will discuss the disk-aware design
optimizations that FFS uses to improve its performance.

There are three key data structures in any file system.
- Superblock stores the information relevant to the file system as a whole. This includes things like the total size of the file system, the block size (i.e., the granularity of a read and write), and which data blocks and inodes (discussed below) are available for use.
- Inodes hold the information or data about a file, i.e., the file metadata. This includes things like the file name, size of the file, which blocks hold the file data, and when was it last accessed or modified.
- Data blocks, as the name suggests, hold the data belonging to the files in the file system. One thing to note is that directories are treated just as files whose data is a list of other files (which could be directories themselves) in the directory.

<figure class="caption"> <img
src="https://github.com/rajatkateja/after-hours-academic/blob/main/images/inode-direct-indirect-pointers.png?raw=true"/> <figcaption>A
common design for storing a file’s data blocks is via direct and indirect
pointers. Direct pointers in blue point to data blocks containing the file’s
data in pink. Indirect pointers are in green and point to blocks of direct
pointers. Double indirect pointers are in orange pointing to blocks of indirect
pointers, and triple indirect pointers are in red pointing to blocks of double
indirect pointers. Having indirect pointers help scale the file (and its number
        of data blocks). For example, consider a block that can hold 4 pointers. With
all direct pointers, it can point to 4 data blocks, but with 1 direct pointer
it can point to 7 data blocks (3 direct, and 4 via the indirect block).
</figcaption> </figure>

A common design for locating a file’s data is to store direct and indirect
pointers to the data blocks in the file’s inode. Direct pointers point to the
file’s data blocks. Indirect pointers point to blocks that contain the direct
pointers to file’s data blocks. Inodes can also have double- or triple-indirect
pointers, i.e., pointers that point to blocks of indirect or double-indirect
pointers respectively. The insight behind having these levels of indirections
is to allow file sizes to grow (increasing data blocks) without a linear
increase in the inode size. If only direct pointers were allowed, the inode
size would have to scale linearly with the file size. This design of direct and
indirect pointers is used by OFS as well as FFS.

<figure class="caption"> <img src="https://github.com/rajatkateja/after-hours-academic/blob/main/images/oldfs-layout.png?raw=true"/>
<figcaption>This image shows the layout of the old file system (OFS) —
superblock, inodes, and data blocks laid out sequentially. The second image
shows one of the performance problems with OFS’s layout — a file’s inode and
data can be far apart on the disk, requiring costly seek operations. The third
image shows another problem with OFS’s layout — over time, because of file
allocations and deletions, the free blocks can be fragmented across the disk.
When a new file is allocated using these fragmented free blocks, its data is
spread across the disk and accessing it requires expensive seek operations.
</figcaption> </figure>

In the old file system, these three data structures (super block, inodes, and
        data blocks) are arranged consecutively. This made the old file system’s design
simple, but the simplicity came at the cost performance downsides because the
design did not take into account the geometry and access mechanism of HDDs.

- A file’s inode and data blocks can be far apart on the disk. This meant that the disk would have to do a costly seek to access the file’s data after accessing the file’s inode (to identify where the file’s data is).
- There was no attempt to locate the inodes of files belonging to the same directory close-by. This meant that any operation that required accessing all the inodes in a directory (e.g., listing all files in a directory) could also incur seeks.
- As the old file system was used over time with file allocations and deletions, the list of its free blocks (called the free list) got fragmented, i.e., the free blocks were spread over the disk’s address space into small groups. This meant that a newly allocated file’s blocks were spread all over the disk and accessing the file sequentially would lead to random accesses on the disk, hurting performance.
- Lastly, the old file system used a block size of 512 bytes, which limited its throughput. Larger block sizes improve throughput because they amortize the cost of the seeking of the disk head to the required track and block.


Now that we understand the performance pain points in the old file system, let
us take a look at how FFS addresses those. At its crux, the performance
improvement in the FFS comes from leveraging the higher sequential throughput
of HDDs. The first optimization that FFS uses is of larger block sizes, e.g.,
   4KB, to improve the access throughput.

   <figure class="caption"> <img src="https://github.com/rajatkateja/after-hours-academic/blob/main/images/hdd-cyclinders.png?raw=true"/>
   <figcaption>Cylinder groups, as shown in the OSTEP chapter on fast file system.
   The diagram shows platter with tracks. The outermost tracks (in dark gray)
    across all the platters form a cylinder. A group of cylinder (here, 3 outermost
            ones) form a cylinder group. </figcaption> </figure>


    To further leverage superior sequential access throughput of HDDs, FFS
    introduced the concept of cylinder groups. Recall that a cylinder is defined as
    the set of the same track across all the platters in the disk. A cylinder group
    is a group of nearby cylinders. In FFS, each cylinder group has a copy of the
    superblock, a group of inodes, and a group of data blocks, arranged
    sequentially. You can think of it as each cylinder group having a miniature (in
            terms of the total data size) old file system. The interesting result is that
    accessing data within this miniature old file system is fast because it lies
    entirely in a cylinder group and seeks within a cylinder group are fast (the
            tracks are close by). Operations like accessing a file’s inode and data, or
    sequentially accessing a file’s data do not incur costly seek operations.


    <figure class="caption"> <img src="
    https://github.com/rajatkateja/after-hours-academic/blob/main/images/ffs-layout.png?raw=true"/>
    <figcaption>An schematic diagram of the fast file system (FFS) layout. It shows
    4 cylinder groups. Each cylinder group has a copy of the file system’s super
    block. Each group has its own set of inodes and data blocks. Essentially, each
    cylinder group has its own mini-OFS. This addresses the key performance problem
    of OFS that it required costly seek operations. Each mini-OFS instance is
    within a cylinder group and seeks within a cylinder group are fast. FFS also
    tries to allocated related data (e.g., files in the same directory) to the same
    cylinder group while also trying to load-balance across cylinder groups (e.g.,
            in its allocation of directories to cylinder groups).  </figcaption> </figure>


    FFS’s policy for allocating files and directories across the cylinder groups
    tries to ensure that related data is not spread randomly across the disk. FFS
    tries to allocate all the inodes of files in the same directory to the same
    cylinder group. To load-balance across cylinder groups, FFS tries to spread out
    the allocation of directories (and hence the files belonging to those
            directories) across the cylinder groups. When allocating data blocks for a
    file, FFS tries to allocate them in a single cylinder group.  The policy of
    localizing file data blocks to a cylinder group poses a challenge with large
    files. A large file can take up most of the available data blocks in a cylinder
    group, forcing other small files in the directory to have their data blocks be
    located in other cylinder groups. To address this problem, after allocating a
    certain number of data blocks for a file, FFS starts allocating subsequent data
    blocks in other cylinder groups (chosen to load-balance the data). With large
    enough thresholds for such spill-overs, the cost of the seeks across cylinder
    groups is amortized over the amount of data accessed within each cylinder
    group. In the original FFS design, the first spill-over happened when the file
    required its first indirect pointer, an intuitive point for a spill-over.
    Subsequent spill-overs happened when a file’s data blocks occupied more than
    25% of the data blocks in a cylinder group — a rather arbitrary choice that
    worked well nevertheless :).  With larger block sizes and the introduction of
    cylinder groups, FFS addressed all of the performance bottlenecks of OFS listed
    above. In addition to the performance enhancements, FFS also introduced new
    functionalities, namely long file names (up to 255 characters long), file
    locking for concurrent accesses (using reader-writer locks), symbolic links,
    user quotas, and the rename system call. All of these functionalities are
    standard in any modern file system. Indeed, FFS paved the way for all future
    file systems in terms of functionalities as well as performance — many modern
    HDD file systems (e.g., ext4, a widely popular Linux file system) use block
    groups akin to cylinder groups, and almost all file systems try to be aware of
    and leverage the peculiarities and idiosyncrasies or the underlying storage
    medium.
