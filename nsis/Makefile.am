AUTOMAKE_OPTIONS = subdir-objects

all:

if MINGW

gitrev="$(shell git --git-dir=${abs_top_srcdir}/.git --work-tree=${abs_top_srcdir} describe --always --tags | sed s/^v//)"

.PHONY: winsetup

Plugins/x86-unicode/INetC.dll:
	curl -OsS https://nsis.sourceforge.io/mediawiki/images/c/c9/Inetc.zip
	unzip Inetc.zip $@

winpath_rc.o: $(top_srcdir)/src/tesseract.rc
	$(WINDRES) -DTESS_RC_PRODUCT_VERSION=$(PACKAGE_VERSION:v%=%) -I$(top_builddir)/include -i $(top_srcdir)/src/tesseract.rc -o $@

winpath.exe: winpath.cpp winpath_rc.o
	$(CXX) -Os -o $@ $< winpath_rc.o
	$(STRIP) --strip-unneeded $@

winsetup: Plugins/x86-unicode/INetC.dll winpath.exe
	case "$(host_cpu)" in \
	aarch64) arch="-DW64 -DARM64";; \
	x86_64) arch="-DW64";; \
	*) arch="";; \
	esac; \
	makensis -DCROSSBUILD -DSHARED -DSIGNCODE=$(SIGNCODE) -DSRCDIR=$(top_srcdir) -DPREFIX=$(prefix) -DVERSION=${gitrev} $$arch -NOCD $(top_srcdir)/nsis/tesseract.nsi

endif
