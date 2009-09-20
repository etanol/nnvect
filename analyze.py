#!/usr/bin/env python

class Info:
        def __init__ (this):
                this.lines   = 0
                this.indices = 0
                this.floats  = False
                this.maximum = 0
                this.minimum = 0


def analyze (path):
        blas = Info()
        file = open(path, "r")
        for line in file:
                blas.lines += 1
                for pair in line.split()[1:]:
                        try:
                                stridx, strval = pair.split(":", 1)
                                index = int(stridx)
                        except ValueError, e:
                                e.blasline = blas.lines
                                file.close()
                                raise e
                        if blas.floats:
                                value = float(strval)
                        else:
                                try:
                                        value = long(strval)
                                except ValueError:
                                        value = float(strval)
                                        blas.maximum = float(blas.maximum)
                                        blas.minimum = float(blas.minimum)
                                        blas.floats = True
                        if index > blas.indices:
                                blas.indices = index
                        if value > blas.maximum:
                                blas.maximum = value
                        elif value < blas.minimum:
                                blas.minimum = value
        file.close()
        return blas


def save (blas, path):
        output = open(path, "w")
        output.write("lines    %s\n" % blas.lines)
        output.write("indices  %s\n" % blas.indices)
        if blas.floats:
                output.write("floats   yes\n")
                output.write("maximum  %lf\n" % blas.maximum)
                output.write("minimum  %lf\n" % blas.minimum)
        else:
                output.write("floats   no\n")
                output.write("maximum  %ld\n" % blas.maximum)
                output.write("minimum  %ld\n" % blas.minimum)
        output.close()



####################################  MAIN  ####################################

import sys
import os, os.path

if len(sys.argv) <= 1:
        print "Please specify the files or directories to analyze"
        sys.exit(1)

filelist = set()
for argument in sys.argv[1:]:
        if os.path.isdir(argument):
                for dent in os.walk(argument):
                        for f in dent[2]:
                                if f.endswith(".tst") or f.endswith(".trn"):
                                        filelist.add(os.path.join(dent[0], f[:-4]))
        elif os.path.isfile(argument + ".tst") and os.path.isfile(argument + ".trn"):
                filelist.add(argument)
        else:
                print "Ignoring %s" % argument

print "Examining %d pairs" % len(filelist)
for path in filelist:
        if os.path.isfile(path + ".tst.info") and \
           os.path.getmtime(path + ".tst.info") > os.path.getmtime(path + ".tst"):
                print "Skipping %s" % path
                continue

        try:
                print "Analyzing %s.tst" % path
                blas = analyze(path + ".tst")
                print "Analyzing %s.trn" % path
                trblas = analyze(path + ".trn")
                blas.indices = max(trblas.indices, blas.indices)
                blas.floats = blas.floats or trblas.floats
                trblas.indices = blas.indices
                trblas.floats = blas.floats
        except OSError, e:
                print "ERROR: Could not open %s: %s" % (path, e)
                continue
        except ValueError, e:
                print "ERROR: Syntax error at %s:%d: %s" % (path, e.blasline, e)
                continue

        try:
                save(blas, path + ".tst.info")
                save(trblas, path + ".trn.info")
        except OSError, e:
                print "ERROR: System error when saving %s: %s" % (path + Info.SUFFIX, e)
        except IOError, e:
                print "ERROR: I/O error when saving %s: %s" % (path + Info.SUFFIX, e)


