AUTOMAKE_OPTIONS = subdir-objects

if MINGW

gitrev="`git --git-dir=${abs_top_srcdir}/.git --work-tree=${abs_top_srcdir} describe --always --tags`"

NSISFLAGS=-DSRCDIR=$(top_srcdir) -DVERSION=${gitrev} -NOCD

ifneq ($(build_os), $(host_os))
NSISFLAGS+=-DCROSSBUILD
fi

#~ if enable_shared
NSISFLAGS+=-DSHARED
#~ endif

if ENABLE_TRAINING
NSISFLAGS+=-DTRAINING
endif

all winsetup:
	makensis $(NSISFLAGS) $(top_srcdir)/nsis/tesseract.nsi

endif
