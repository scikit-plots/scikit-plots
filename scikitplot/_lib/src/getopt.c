

// https://www.man7.org/linux/man-pages/man3/getopt.3.html
// getopt(int argc, char *const argv[], const char *optstring)
// getopt() function in C to parse command line arguments.
// optstring is simply  a list of characters, each representing a single character option.


// Program to illustrate the getopt()
// function in C

#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int opt;

    // put ':' in the starting of the
    // string so that program can
    //distinguish between '?' and ':'
    while((opt = getopt(argc, argv, “:if:lrx”)) != -1)
    {
        switch(opt)
        {
            case ‘i’:
            case ‘l’:
            case ‘r’:
                printf(“option: %c\n”, opt);
                break;
            case ‘f’:
                printf(“filename: %s\n”, optarg);
                break;
            case ‘:’:
                printf(“option needs a value\n”);
                break;
            case ‘?’:
                printf(“unknown option: %c\n”, optopt);
                break;
        }
    }

    // optind is for the extra arguments
    // which are not parsed
    for(; optind < argc; optind++){
        printf(“extra arguments: %s\n”, argv[optind]);
    }

    return 0;
}
