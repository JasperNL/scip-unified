#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2002-2016 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic Licence.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#@file    Makefile
#@brief   SCIP Makefile
#@author  Thorsten Koch
#@author  Tobias Achterberg
#@author  Marc Pfetsch
#@author  Timo Berthold

#-----------------------------------------------------------------------------
# paths variables
#-----------------------------------------------------------------------------

SCIPDIR		=	./

INSTALLDIR	=

#-----------------------------------------------------------------------------
# include make.project file
#-----------------------------------------------------------------------------

# do not use other open source projects; needs to be set before including make.project
ifeq ($(OPENSOURCE),false)
	override EXPRINT	=	none
	override GMP		=	false
	override READLINE	=	false
	override ZLIB		=	false
	override ZIMPL		=	false
	override IPOPT		=	false
endif

# mark that this is a SCIP internal makefile
SCIPINTERNAL	=	true

# load default settings and detect host architecture
include $(SCIPDIR)/make/make.project

#-----------------------------------------------------------------------------
# define build flags
#-----------------------------------------------------------------------------
BUILDFLAGS =	" ARCH=$(ARCH)\\n\
		COMP=$(COMP)\\n\
		EXPRINT=$(EXPRINT)\\n\
		GAMS=$(GAMS)\\n\
		GMP=$(GMP)\\n\
		IPOPT=$(IPOPT)\\n\
		IPOPTOPT=$(IPOPTOPT)\\n\
		LPS=$(LPS)\\n\
		LPSCHECK=$(LPSCHECK)\\n\
		LPSOPT=$(LPSOPT)\\n\
		NOBLKBUFMEM=$(NOBLKBUFMEM)\\n\
		NOBLKMEM=$(NOBLKMEM)\\n\
		NOBUFMEM=$(NOBUFMEM)\\n\
		OPT=$(OPT)\\n\
		OSTYPE=$(OSTYPE)\\n\
		PARASCIP=$(PARASCIP)\\n\
		READLINE=$(READLINE)\\n\
		SANITIZE=$(SANITIZE)\\n\
		SHARED=$(SHARED)\\n\
		USRARFLAGS=$(USRARFLAGS)\\n\
		USRCFLAGS=$(USRCFLAGS)\\n\
		USRCXXFLAGS=$(USRCXXFLAGS)\\n\
		USRDFLAGS=$(USRDFLAGS)\\n\
		USRFLAGS=$(USRFLAGS)\\n\
		USRLDFLAGS=$(USRLDFLAGS)\\n\
		USROFLAGS=$(USROFLAGS)\\n\
		VERSION=$(VERSION)\\n\
		ZIMPL=$(ZIMPL)\\n\
		ZIMPLOPT=$(ZIMPLOPT)\\n\
		ZLIB=$(ZLIB)"

#-----------------------------------------------------------------------------
# default settings
#-----------------------------------------------------------------------------

VERSION		=	3.2.1.2
SCIPGITHASH	=
SOFTLINKS	=
MAKESOFTLINKS	=	true
TOUCHLINKS	=	false

#-----------------------------------------------------------------------------
# LP Solver Interface
#-----------------------------------------------------------------------------

LPILIBSHORTNAME	=	lpi$(LPS)
LPILIBNAME	=	$(LPILIBSHORTNAME)-$(VERSION)
LPILIBOBJ	=
LPSOPTIONS	=
LPIINSTMSG	=

LPSCHECKDEP	:=	$(SRCDIR)/depend.lpscheck
LPSCHECKSRC	:=	$(shell cat $(LPSCHECKDEP))

LPSOPTIONS	+=	cpx
ifeq ($(LPS),cpx)
FLAGS		+=	-I$(LIBDIR)/include/cpxinc
LPILIBOBJ	=	lpi/lpi_cpx.o scip/bitencode.o blockmemshell/memory.o scip/message.o
LPILIBSRC  	=	$(addprefix $(SRCDIR)/,$(LPILIBOBJ:.o=.c))
SOFTLINKS	+=	$(LIBDIR)/include/cpxinc
ifeq ($(SHARED),true)
SOFTLINKS	+=	$(LIBDIR)/shared/libcplex.$(OSTYPE).$(ARCH).$(COMP).$(SHAREDLIBEXT)
else
SOFTLINKS	+=	$(LIBDIR)/static/libcplex.$(OSTYPE).$(ARCH).$(COMP).$(STATICLIBEXT)
endif
LPIINSTMSG	=	"  -> \"cpxinc\" is the path to the CPLEX \"include\" directory, e.g., \"<CPLEX-path>/include/ilcplex\".\n"
LPIINSTMSG	+=	" -> \"libcplex.*.a\" is the path to the CPLEX library, e.g., \"<CPLEX-path>/lib/x86-64_linux/static_pic/libcplex.a\"\n"
LPIINSTMSG	+=	" -> \"libcplex.*.so\" is the path to the CPLEX library, e.g., \"<CPLEX-path>/bin/x86-64_linux/libcplex1263.so\""
endif

LPSOPTIONS	+=	xprs
ifeq ($(LPS),xprs)
FLAGS		+=	-I$(LIBDIR)/include/xprsinc
LPILIBOBJ	=	lpi/lpi_xprs.o scip/bitencode.o blockmemshell/memory.o scip/message.o
LPILIBSRC  	=	$(addprefix $(SRCDIR)/,$(LPILIBOBJ:.o=.c))
SOFTLINKS	+=	$(LIBDIR)/include/xprsinc
ifeq ($(SHARED),true)
SOFTLINKS	+=	$(LIBDIR)/shared/libxpress.$(OSTYPE).$(ARCH).$(COMP).$(SHAREDLIBEXT)
else
SOFTLINKS	+=	$(LIBDIR)/static/libxpress.$(OSTYPE).$(ARCH).$(COMP).$(STATICLIBEXT)
endif
LPIINSTMSG	=	"  -> \"xprsinc\" is the path to the XPRESS \"include\" directory, e.g., \"<XPRESS-path>/include\".\n"
LPIINSTMSG	+=	" -> \"libpress.*\" is the path to the XPRESS library, e.g., \"<XPRESS-path>/lib/libxpress.a\""
endif

LPSOPTIONS	+=	msk
ifeq ($(LPS),msk)
FLAGS		+=	-I$(LIBDIR)/include/mskinc
LPILIBOBJ	=	lpi/lpi_msk.o scip/bitencode.o blockmemshell/memory.o scip/message.o
LPILIBSRC  	=	$(addprefix $(SRCDIR)/,$(LPILIBOBJ:.o=.c))
SOFTLINKS	+=	$(LIBDIR)/include/mskinc
ifeq ($(SHARED),true)
SOFTLINKS	+=	$(LIBDIR)/shared/libmosek.$(OSTYPE).$(ARCH).$(COMP).$(SHAREDLIBEXT)
SOFTLINKS	+=	$(LIBDIR)/shared/libiomp5.$(OSTYPE).$(ARCH).$(COMP).$(SHAREDLIBEXT)
else
SOFTLINKS	+=	$(LIBDIR)/static/libmosek.$(OSTYPE).$(ARCH).$(COMP).$(STATICLIBEXT)
SOFTLINKS	+=	$(LIBDIR)/static/libiomp5.$(OSTYPE).$(ARCH).$(COMP).$(STATICLIBEXT)
endif
LPIINSTMSG	=	"  -> \"mskinc\" is the path to the Mosek \"include\" directory, e.g., \"<Mosek-path>/include\".\n"
LPIINSTMSG	+=	" -> \"libmosek.*\" is the path to the Mosek library, e.g., \"<Mosek-path>/lib/libmosek.a\".\n"
LPIINSTMSG	+=	" -> \"libiomp5.*\" is the path to the libiomp5, e.g., \"<Mosek-path>/lib/libiomp5.a\""
endif

LPSOPTIONS	+=	spx1
ifeq ($(LPS),spx1)
LINKER		=	CPP
FLAGS		+=	-I$(LIBDIR)/include/spxinc
ifeq ($(SPX_LEGACY),true)
CFLAGS		+= 	-DSOPLEX_LEGACY
CXXFLAGS	+= 	-DSOPLEX_LEGACY
endif
LPILIBOBJ	=	lpi/lpi_spx1.o scip/bitencode.o blockmemshell/memory.o scip/message.o
LPILIBSRC	=	$(SRCDIR)/lpi/lpi_spx1.cpp $(SRCDIR)/scip/bitencode.c $(SRCDIR)/blockmemshell/memory.c $(SRCDIR)/scip/message.c
SOFTLINKS	+=	$(LIBDIR)/include/spxinc
ifeq ($(SHARED),true)
SOFTLINKS	+=	$(LIBDIR)/shared/libsoplex.$(OSTYPE).$(ARCH).$(COMP).$(LPSOPT).$(SHAREDLIBEXT)
else
SOFTLINKS	+=	$(LIBDIR)/static/libsoplex.$(OSTYPE).$(ARCH).$(COMP).$(LPSOPT).$(STATICLIBEXT)
endif
LPIINSTMSG	=	"  -> \"spxinc\" is the path to the SoPlex \"src\" directory, e.g., \"<SoPlex-path>/src\".\n"
LPIINSTMSG	+=	" -> \"libsoplex.*\" is the path to the SoPlex library, e.g., \"<SoPlex-path>/lib/libsoplex.$(OSTYPE).$(ARCH).$(COMP).$(LPSOPT).$(STATICLIBEXT)\""
ifeq ($(LPSCHECK),true)
FLAGS		+=	-DWITH_LPSCHECK -I$(LIBDIR)/cpxinc
SOFTLINKS	+=	$(LIBDIR)/include/cpxinc
ifeq ($(SHARED),true)
SOFTLINKS	+=	$(LIBDIR)/shared/libcplex.$(OSTYPE).$(ARCH).$(COMP).$(SHAREDLIBEXT)
else
SOFTLINKS	+=	$(LIBDIR)/static/libcplex.$(OSTYPE).$(ARCH).$(COMP).$(STATICLIBEXT)
endif
LPIINSTMSG	+=	"  -> \"cpxinc\" is the path to the CPLEX \"include\" directory, e.g., \"<CPLEX-path>/include/ilcplex\".\n"
LPIINSTMSG	+=	" -> \"libcplex.*.a\" is the path to the CPLEX library, e.g., \"<CPLEX-path>/lib/x86-64_linux/static_pic/libcplex.a\"\n"
LPIINSTMSG	+=	" -> \"libcplex.*.so\" is the path to the CPLEX library, e.g., \"<CPLEX-path>/bin/x86-64_linux/libcplex1263.so\""
endif
endif

LPSOPTIONS	+=	spx ( = spx2)
ifeq ($(LPS),spx2)
LINKER		=	CPP
FLAGS		+=	-I$(LIBDIR)/include/spxinc
LPILIBOBJ	=	lpi/lpi_spx2.o scip/bitencode.o blockmemshell/memory.o scip/message.o
LPILIBSRC	=	$(SRCDIR)/lpi/lpi_spx2.cpp $(SRCDIR)/scip/bitencode.c $(SRCDIR)/blockmemshell/memory.c $(SRCDIR)/scip/message.c
SOFTLINKS	+=	$(LIBDIR)/include/spxinc
ifeq ($(SHARED),true)
SOFTLINKS	+=	$(LIBDIR)/shared/libsoplex.$(OSTYPE).$(ARCH).$(COMP).$(LPSOPT).$(SHAREDLIBEXT)
else
SOFTLINKS	+=	$(LIBDIR)/static/libsoplex.$(OSTYPE).$(ARCH).$(COMP).$(LPSOPT).$(STATICLIBEXT)
endif
LPIINSTMSG	=	"  -> \"spxinc\" is the path to the SoPlex \"src\" directory, e.g., \"<SoPlex-path>/src\".\n"
LPIINSTMSG	+=	" -> \"libsoplex.*\" is the path to the SoPlex library, e.g., \"<SoPlex-path>/lib/libsoplex.linux.x86.gnu.opt.a\""
ifeq ($(LPSCHECK),true)
FLAGS		+=	-DWITH_LPSCHECK -I$(LIBDIR)/cpxinc
SOFTLINKS	+=	$(LIBDIR)/include/cpxinc
ifeq ($(SHARED),true)
SOFTLINKS	+=	$(LIBDIR)/shared/libcplex.$(OSTYPE).$(ARCH).$(COMP).$(SHAREDLIBEXT)
else
SOFTLINKS	+=	$(LIBDIR)/static/libcplex.$(OSTYPE).$(ARCH).$(COMP).$(STATICLIBEXT)
endif
LPIINSTMSG	+=	"  -> \"cpxinc\" is the path to the CPLEX \"include\" directory, e.g., \"<CPLEX-path>/include/ilcplex\".\n"
LPIINSTMSG	+=	" -> \"libcplex.*.a\" is the path to the CPLEX library, e.g., \"<CPLEX-path>/lib/x86_rhel4.0_3.4/static_pic/libcplex.a\"\n"
LPIINSTMSG	+=	" -> \"libcplex.*.so\" is the path to the CPLEX library, e.g., \"<CPLEX-path>/bin/x86-64_linux/libcplex1263.so\""
endif
endif

LPSOPTIONS	+=	clp
ifeq ($(LPS),clp)
LINKER		=	CPP
FLAGS		+=	-I$(LIBDIR)/clp.$(OSTYPE).$(ARCH).$(COMP).$(LPSOPT)/include/coin
LPILIBOBJ	=	lpi/lpi_clp.o scip/bitencode.o blockmemshell/memory.o scip/message.o
LPILIBSRC	=	$(SRCDIR)/lpi/lpi_clp.cpp $(SRCDIR)/scip/bitencode.c $(SRCDIR)/blockmemshell/memory.c $(SRCDIR)/scip/message.c
SOFTLINKS	+=	$(LIBDIR)/clp.$(OSTYPE).$(ARCH).$(COMP).$(LPSOPT)
LPIINSTMSG	=	"  -> \"clp.$(OSTYPE).$(ARCH).$(COMP).$(LPSOPT)\" is the path to the Clp installation directory, i.e., \"<Clp-path>/include/coin/ClpModel.hpp\" should exist.\n"
endif

LPSOPTIONS	+=	qso
ifeq ($(LPS),qso)
FLAGS         	+=      -I$(LIBDIR)/include/qsinc
LPILIBOBJ     	= 	lpi/lpi_qso.o scip/bitencode.o blockmemshell/memory.o scip/message.o
LPILIBSRC     	=       $(addprefix $(SRCDIR)/,$(LPILIBOBJ:.o=.c))
SOFTLINKS     	+=      $(LIBDIR)/include/qsinc
SOFTLINKS     	+=      $(LIBDIR)/static/libqsopt.$(OSTYPE).$(ARCH).$(COMP).$(STATICLIBEXT)
LPIINSTMSG	=	"  -> \"qsinc\" is the path to the QSopt \"include\" directory, e.g., \"<QSopt-path>\".\n"
LPIINSTMSG	+=	" -> \"libqsopt.*\" is the path to the QSopt library, e.g., \"<QSopt-path>/libqsopt.a\""
endif

LPSOPTIONS	+=	grb
ifeq ($(LPS),grb)
FLAGS		+=	-I$(LIBDIR)/include/grbinc
LPILIBOBJ	=	lpi/lpi_grb.o blockmemshell/memory.o scip/message.o
LPILIBSRC  	=	$(addprefix $(SRCDIR)/,$(LPILIBOBJ:.o=.c))
SOFTLINKS	+=	$(LIBDIR)/include/grbinc
ifeq ($(SHARED),true)
SOFTLINKS	+=	$(LIBDIR)/shared/libgurobi.$(OSTYPE).$(ARCH).$(COMP).$(SHAREDLIBEXT)
else
SOFTLINKS	+=	$(LIBDIR)/static/libgurobi.$(OSTYPE).$(ARCH).$(COMP).$(STATICLIBEXT)
endif
LPIINSTMSG	=	"  -> \"grbinc\" is the path to the Gurobi \"include\" directory, e.g., \"<Gurobi-path>/include\".\n"
LPIINSTMSG	+=	" -> \"libgurobi.*\" is the path to the Gurobi library, e.g., \"<Gurobi-path>/lib/libgurobi.so\""
endif

LPSOPTIONS	+=	none
ifeq ($(LPS),none)
LPILIBOBJ	=	lpi/lpi_none.o blockmemshell/memory.o scip/message.o
LPILIBSRC  	=	$(addprefix $(SRCDIR)/,$(LPILIBOBJ:.o=.c))
endif

LPILIB		=	$(LPILIBNAME).$(BASE)
LPILIBFILE	=	$(LIBDIR)/$(LIBTYPE)/lib$(LPILIB).$(LIBEXT)
LPILIBOBJFILES	=	$(addprefix $(LIBOBJDIR)/,$(LPILIBOBJ))
LPILIBDEP	=	$(SRCDIR)/depend.lpilib.$(LPS).$(OPT)
LPILIBLINK	=	$(LIBDIR)/$(LIBTYPE)/lib$(LPILIBSHORTNAME).$(BASE).$(LIBEXT)
LPILIBSHORTLINK = 	$(LIBDIR)/$(LIBTYPE)/lib$(LPILIBSHORTNAME).$(LIBEXT)

ifeq ($(SHARED),true)
LPILIBEXTLIBS	=	$(LIBBUILD_L)$(LIBDIR)/$(LIBTYPE) $(LPSLDFLAGS) $(LINKRPATH)$(realpath $(LIBDIR)/$(LIBTYPE))
endif


#-----------------------------------------------------------------------------
# NLP Solver Interfaces and expression interpreter
#-----------------------------------------------------------------------------

NLPILIBCOBJ	= 	nlpi/nlpi.o \
			nlpi/nlpioracle.o \
			nlpi/expr.o

NLPILIBCXXOBJ	= 	nlpi/intervalarithext.o

NLPILIBSCIPOBJ	= 	blockmemshell/memory.o \
			scip/misc.o \
			scip/intervalarith.o \
			scip/interrupt.o \
			scip/message.o

ifeq ($(EXPRINT),none)
NLPILIBCOBJ 	+=	nlpi/exprinterpret_none.o
endif
ifeq ($(EXPRINT),cppad)
NLPILIBCXXOBJ 	+= 	nlpi/exprinterpret_cppad.o
endif

ifeq ($(IPOPT),true)
NLPILIBSHORTNAME = $(NLPILIBSHORTNAME).ipopt
NLPILIBCXXOBJ	+= 	nlpi/nlpi_ipopt.o
else
NLPILIBCOBJ	+= 	nlpi/nlpi_ipopt_dummy.o
endif

NLPILIBSHORTNAME= 	nlpi$(NLPILIBSHORTNAMECPPAD)$(NLPILIBSHORTNAMEIPOPT)
NLPILIBNAME	=	$(NLPILIBSHORTNAME)-$(VERSION)
NLPILIB		=	$(NLPILIBNAME).$(BASE)
NLPILIBFILE	=	$(LIBDIR)/$(LIBTYPE)/lib$(NLPILIB).$(LIBEXT)
NLPILIBOBJFILES =	$(addprefix $(LIBOBJDIR)/,$(NLPILIBCOBJ)) $(addprefix $(LIBOBJDIR)/,$(NLPILIBCXXOBJ))
NLPILIBSCIPOBJFILES =	$(addprefix $(LIBOBJDIR)/,$(NLPILIBSCIPOBJ))
NLPILIBSRC	=	$(addprefix $(SRCDIR)/,$(NLPILIBCOBJ:.o=.c)) $(addprefix $(SRCDIR)/,$(NLPILIBCXXOBJ:.o=.cpp))
NLPILIBDEP	=	$(SRCDIR)/depend.nlpilib$(NLPILIBSHORTNAMECPPAD)$(NLPILIBSHORTNAMEIPOPT).$(OPT)
NLPILIBLINK	=	$(LIBDIR)/$(LIBTYPE)/lib$(NLPILIBSHORTNAME).$(BASE).$(LIBEXT)
NLPILIBSHORTLINK=	$(LIBDIR)/$(LIBTYPE)/lib$(NLPILIBSHORTNAME).$(LIBEXT)

ifeq ($(SHARED),true)
NLPILIBEXTLIBS	=	$(LIBBUILD_L)$(LIBDIR)/$(LIBTYPE) $(IPOPTLIBS) \
			$(LINKRPATH)$(realpath $(LIBDIR)/$(LIBTYPE)/ipopt.$(OSTYPE).$(ARCH).$(COMP).$(IPOPTOPT)/lib)
endif


#-----------------------------------------------------------------------------
# External Libraries
#-----------------------------------------------------------------------------

ZLIBDEP		:=	$(SRCDIR)/depend.zlib
ZLIBSRC		:=	$(shell cat $(ZLIBDEP))

GMPDEP		:=	$(SRCDIR)/depend.gmp
GMPSRC		:=	$(shell cat $(GMPDEP))

READLINEDEP	:=	$(SRCDIR)/depend.readline
READLINESRC	:=	$(shell cat $(READLINEDEP))

ZIMPLDEP	:=	$(SRCDIR)/depend.zimpl
ZIMPLSRC	:=	$(shell cat $(ZIMPLDEP))

GAMSDEP		:=	$(SRCDIR)/depend.gams
GAMSSRC		:=	$(shell cat $(GAMSDEP))

PARASCIPDEP	:=	$(SRCDIR)/depend.parascip
PARASCIPSRC	:=	$(shell cat $(PARASCIPDEP))

ifeq ($(ZIMPL),true)
ifeq ($(GMP),false)
$(error ZIMPL requires the GMP to be linked. Use either ZIMPL=false or GMP=true.)
endif
FLAGS		+=	-DWITH_ZIMPL -I$(LIBDIR)/include/zimplinc $(ZIMPL_FLAGS)
DIRECTORIES	+=	$(LIBDIR)/include/zimplinc
SOFTLINKS	+=	$(LIBDIR)/include/zimplinc/zimpl
ifeq ($(SHARED),true)
SOFTLINKS	+=	$(LIBDIR)/shared/libzimpl.$(OSTYPE).$(ARCH).$(COMP).$(ZIMPLOPT).$(SHAREDLIBEXT)
else
SOFTLINKS	+=	$(LIBDIR)/static/libzimpl.$(OSTYPE).$(ARCH).$(COMP).$(ZIMPLOPT).$(STATICLIBEXT)
endif
LPIINSTMSG	+=	"\n  -> \"zimplinc\" is a directory containing the path to the ZIMPL \"src\" directory, e.g., \"<ZIMPL-path>/src\".\n"
LPIINSTMSG	+=	" -> \"libzimpl.*\" is the path to the ZIMPL library, e.g., \"<ZIMPL-path>/lib/libzimpl.$(OSTYPE).$(ARCH).$(COMP).$(ZIMPLOPT).$(STATICLIBEXT)\""
endif

ifeq ($(GMP),true)
ifeq ($(COMP),msvc)
SOFTLINKS	+=	$(LIBDIR)/mpir.$(ARCH)
SOFTLINKS	+=	$(LIBDIR)/$(LIBTYPE)/libmpir.$(ARCH).$(OPT).lib
SOFTLINKS	+=	$(LIBDIR)/$(LIBTYPE)/libpcre.$(ARCH).$(OPT).lib
LPIINSTMSG	+=	"\n  -> \"mpir.$(ARCH)\" is a directory containing the mpir installation, i.e., \"mpir.$(ARCH)/gmp.h\" should exist.\n"
LPIINSTMSG	+=	" -> \"libmpir.*\" is the path to the MPIR library\n"
LPIINSTMSG	+=	" -> \"libpcre.*\" is the path to the PCRE library"
endif
endif

ifeq ($(IPOPT),true)
SOFTLINKS	+=	$(LIBDIR)/$(LIBTYPE)/ipopt.$(OSTYPE).$(ARCH).$(COMP).$(IPOPTOPT)
LPIINSTMSG	+=	"\n  -> \"ipopt.$(OSTYPE).$(ARCH).$(COMP).$(IPOPTOPT)\" is a directory containing the ipopt installation, i.e., \"ipopt.$(OSTYPE).$(ARCH).$(COMP).$(IPOPTOPT)/include/coin/IpIpoptApplication.hpp\", \"ipopt.$(OSTYPE).$(ARCH).$(COMP).$(IPOPTOPT)/lib/libipopt*\", ... should exist.\n"
endif

ifeq ($(GAMS),true)
GAMSDIR		=	$(LIBDIR)/gams.$(OSTYPE).$(ARCH).$(COMP)
FLAGS		+=	-DWITH_GAMS=\"$(abspath $(GAMSDIR))\"
FLAGS		+=	-I$(SCIPDIR)/interfaces/gams/src -I$(GAMSDIR)/apifiles/C/api
SOFTLINKS	+=	$(GAMSDIR)
LPIINSTMSG	+=	"\n  -> \"$(GAMSDIR)\" is the path to the GAMS system directory"
endif

ifeq ($(SHARED),true)
SCIPLIBEXTLIBS	=	$(LIBBUILD_L)$(LIBDIR)/$(LIBTYPE) $(ZLIB_LDFLAGS) $(GMP_LDFLAGS) $(READLINE_LDFLAGS) $(ZIMPLLIB) \
			$(LINKRPATH)$(realpath $(LIBDIR)/$(LIBTYPE))
endif


#-----------------------------------------------------------------------------
# SCIP Library
#-----------------------------------------------------------------------------

SCIPLIBSHORTNAME=	scip
SCIPLIBNAME	=	$(SCIPLIBSHORTNAME)-$(VERSION)
SCIPPLUGINLIBOBJ=       scip/branch_allfullstrong.o \
			scip/branch_cloud.o \
			scip/branch_distribution.o \
			scip/branch_fullstrong.o \
			scip/branch_inference.o \
			scip/branch_leastinf.o \
			scip/branch_mostinf.o \
			scip/branch_multaggr.o \
			scip/branch_nodereopt.o \
			scip/branch_pscost.o \
			scip/branch_random.o \
			scip/branch_relpscost.o \
			scip/cons_abspower.o \
			scip/compr_largestrepr.o \
			scip/compr_weakcompr.o \
			scip/cons_and.o \
			scip/cons_bivariate.o \
			scip/cons_bounddisjunction.o \
			scip/cons_conjunction.o \
			scip/cons_countsols.o \
			scip/cons_cumulative.o \
			scip/cons_disjunction.o \
			scip/cons_indicator.o \
			scip/cons_integral.o \
			scip/cons_knapsack.o \
			scip/cons_linear.o \
			scip/cons_linking.o \
			scip/cons_logicor.o \
			scip/cons_nonlinear.o \
			scip/cons_or.o \
			scip/cons_orbitope.o \
			scip/cons_pseudoboolean.o \
			scip/cons_quadratic.o \
			scip/cons_setppc.o \
			scip/cons_soc.o \
			scip/cons_sos1.o \
			scip/cons_sos2.o \
			scip/cons_superindicator.o \
			scip/cons_varbound.o \
			scip/cons_xor.o \
			scip/dialog_default.o \
			scip/event_softtimelimit.o \
			scip/disp_default.o \
			scip/event_solvingphase.o \
			scip/heur_actconsdiving.o \
			scip/heur_bound.o \
			scip/heur_clique.o \
			scip/heur_coefdiving.o \
			scip/heur_completesol.o \
			scip/heur_crossover.o \
			scip/heur_dins.o \
			scip/heur_distributiondiving.o \
			scip/heur_dualval.o \
			scip/heur_feaspump.o \
			scip/heur_fixandinfer.o \
			scip/heur_fracdiving.o \
			scip/heur_guideddiving.o \
			scip/heur_indicator.o \
			scip/heur_intdiving.o \
			scip/heur_intshifting.o \
			scip/heur_linesearchdiving.o \
			scip/heur_localbranching.o \
			scip/heur_mutation.o \
			scip/heur_multistart.o \
			scip/heur_nlpdiving.o \
			scip/heur_objpscostdiving.o \
			scip/heur_octane.o \
			scip/heur_ofins.o \
			scip/heur_oneopt.o \
			scip/heur_proximity.o \
			scip/heur_pscostdiving.o \
			scip/heur_reoptsols.o \
			scip/heur_randrounding.o \
			scip/heur_rens.o \
			scip/heur_rins.o \
			scip/heur_rootsoldiving.o \
			scip/heur_rounding.o \
			scip/heur_shiftandpropagate.o \
			scip/heur_shifting.o \
			scip/heur_simplerounding.o \
			scip/heur_subnlp.o \
			scip/heur_trivial.o \
			scip/heur_trivialnegation.o \
			scip/heur_trysol.o \
			scip/heur_twoopt.o \
			scip/heur_undercover.o \
			scip/heur_vbounds.o \
			scip/heur_veclendiving.o \
			scip/heur_zeroobj.o \
			scip/heur_zirounding.o \
			scip/message_default.o \
			scip/nodesel_bfs.o \
			scip/nodesel_breadthfirst.o \
			scip/nodesel_dfs.o \
			scip/nodesel_estimate.o \
			scip/nodesel_hybridestim.o \
			scip/nodesel_restartdfs.o \
			scip/nodesel_uct.o \
			scip/presol_boundshift.o \
			scip/presol_components.o \
			scip/presol_convertinttobin.o \
			scip/presol_domcol.o\
			scip/presol_dualagg.o\
			scip/presol_dualcomp.o\
			scip/presol_dualinfer.o\
			scip/presol_gateextraction.o \
			scip/presol_implfree.o\
			scip/presol_implics.o \
			scip/presol_inttobinary.o \
			scip/presol_redvub.o \
			scip/presol_trivial.o \
			scip/presol_tworowbnd.o \
			scip/presol_stuffing.o \
			scip/prop_dualfix.o \
			scip/prop_genvbounds.o \
			scip/prop_obbt.o \
			scip/prop_probing.o \
			scip/prop_pseudoobj.o \
			scip/prop_redcost.o \
			scip/prop_rootredcost.o \
			scip/prop_vbounds.o \
			scip/reader_bnd.o \
			scip/reader_ccg.o \
			scip/reader_cip.o \
			scip/reader_cnf.o \
			scip/reader_diff.o \
			scip/reader_fix.o \
			scip/reader_fzn.o \
			scip/reader_gms.o \
			scip/reader_lp.o \
			scip/reader_mps.o \
			scip/reader_mst.o \
			scip/reader_opb.o \
			scip/reader_osil.o \
			scip/reader_pip.o \
			scip/reader_pbm.o \
			scip/reader_ppm.o \
			scip/reader_rlp.o \
			scip/reader_sol.o \
			scip/reader_wbo.o \
			scip/reader_zpl.o \
			scip/sepa_cgmip.o \
			scip/sepa_clique.o \
			scip/sepa_closecuts.o \
			scip/sepa_cmir.o \
			scip/sepa_disjunctive.o \
			scip/sepa_eccuts.o \
			scip/sepa_flowcover.o \
			scip/sepa_gomory.o \
			scip/sepa_impliedbounds.o \
			scip/sepa_intobj.o \
			scip/sepa_mcf.o \
			scip/sepa_oddcycle.o \
			scip/sepa_rapidlearning.o \
			scip/sepa_strongcg.o \
			scip/sepa_zerohalf.o

SCIPLIBOBJ	=	scip/branch.o \
			scip/clock.o \
			scip/conflict.o \
			scip/cons.o \
			scip/cutpool.o \
			scip/debug.o \
			scip/dialog.o \
			scip/disp.o \
			scip/event.o \
			scip/fileio.o \
			scip/heur.o \
			scip/heuristics.o \
			scip/compr.o \
			scip/history.o \
			scip/implics.o \
			scip/interrupt.o \
			scip/intervalarith.o \
			scip/lp.o \
			scip/matrix.o \
			scip/mem.o \
			scip/misc.o \
			scip/nlp.o \
			scip/nodesel.o \
			scip/paramset.o \
			scip/presol.o \
			scip/presolve.o \
			scip/pricestore.o \
			scip/pricer.o \
			scip/primal.o \
			scip/prob.o \
			scip/prop.o \
			scip/reader.o \
			scip/relax.o \
			scip/reopt.o \
			scip/retcode.o \
			scip/scip.o \
			scip/scipbuildflags.o \
			scip/scipdefplugins.o \
			scip/scipgithash.o \
			scip/scipshell.o \
			scip/sepa.o \
			scip/sepastore.o \
			scip/set.o \
			scip/sol.o \
			scip/solve.o \
			scip/stat.o \
			scip/tree.o \
			scip/var.o \
			scip/visual.o \
			tclique/tclique_branch.o \
			tclique/tclique_coloring.o \
			tclique/tclique_graph.o \
			dijkstra/dijkstra.o \
			xml/xmlparse.o

# the SCIP library contains all files except objscip
SCIPLIB		=	$(SCIPLIBNAME).$(BASE).$(LPS)
SCIPLIBFILE	=	$(LIBDIR)/$(LIBTYPE)/lib$(SCIPLIB).$(LIBEXT)
SCIPLIBOBJFILES	=	$(addprefix $(LIBOBJDIR)/,$(SCIPPLUGINLIBOBJ))
SCIPLIBOBJFILES	+=	$(addprefix $(LIBOBJDIR)/,$(SCIPLIBOBJ))
SCIPLIBOBJFILES +=	$(LPILIBOBJFILES)
SCIPLIBOBJFILES +=	$(NLPILIBOBJFILES)
SCIPPLUGININCSRC=	$(addprefix $(SRCDIR)/,$(SCIPPLUGINLIBOBJ:.o=.h))
SCIPLIBSRC	=	$(addprefix $(SRCDIR)/,$(SCIPPLUGINLIBOBJ:.o=.c))
SCIPLIBSRC	+=	$(addprefix $(SRCDIR)/,$(SCIPLIBOBJ:.o=.c))
SCIPLIBSRC	+=	$(LPILIBSRC)
SCIPLIBSRC	+=	$(NLPILIBSRC)
SCIPLIBDEP	=	$(SRCDIR)/depend.sciplib.$(OPT)
SCIPLIBLINK	=	$(LIBDIR)/$(LIBTYPE)/lib$(SCIPLIBSHORTNAME).$(BASE).$(LPS).$(LIBEXT)
SCIPLIBSHORTLINK = 	$(LIBDIR)/$(LIBTYPE)/lib$(SCIPLIBSHORTNAME).$(LPS).$(LIBEXT)

ifeq ($(GAMS),true)
SCIPLIBOBJFILES += 	$(addprefix $(LIBOBJDIR)/scip/,gmomcc.o gevmcc.o reader_gmo.o)
endif

ALLSRC		=	$(SCIPLIBSRC)

SCIPGITHASHFILE	= 	$(SRCDIR)/scip/githash.c
SCIPBUILDFLAGSFILE = 	$(SRCDIR)/scip/buildflags.c


#-----------------------------------------------------------------------------
# Objective SCIP Library
#-----------------------------------------------------------------------------

OBJSCIPLIBSHORTNAME=	objscip
OBJSCIPLIBNAME	=	$(OBJSCIPLIBSHORTNAME)-$(VERSION)
OBJSCIPLIBOBJ	=	objscip/objbranchrule.o \
			objscip/objconshdlr.o \
			objscip/objdialog.o \
			objscip/objdisp.o \
			objscip/objeventhdlr.o \
			objscip/objheur.o \
			objscip/objmessagehdlr.o \
			objscip/objnodesel.o \
			objscip/objpresol.o \
			objscip/objpricer.o \
			objscip/objprobdata.o \
			objscip/objprop.o \
			objscip/objreader.o \
			objscip/objrelax.o \
			objscip/objsepa.o \
			objscip/objvardata.o

OBJSCIPLIB	=	$(OBJSCIPLIBNAME).$(BASE)
OBJSCIPLIBFILE	=	$(LIBDIR)/$(LIBTYPE)/lib$(OBJSCIPLIB).$(LIBEXT)
OBJSCIPLIBOBJFILES=	$(addprefix $(LIBOBJDIR)/,$(OBJSCIPLIBOBJ))
OBJSCIPLIBSRC	=	$(addprefix $(SRCDIR)/,$(OBJSCIPLIBOBJ:.o=.cpp))
OBJSCIPINCSRC	=	$(addprefix $(SRCDIR)/,$(OBJSCIPLIBOBJ:.o=.h))
OBJSCIPLIBDEP	=	$(SRCDIR)/depend.objsciplib.$(OPT)
OBJSCIPLIBLINK	=	$(LIBDIR)/$(LIBTYPE)/lib$(OBJSCIPLIBSHORTNAME).$(BASE).$(LIBEXT)
OBJSCIPLIBSHORTLINK=	$(LIBDIR)/$(LIBTYPE)/lib$(OBJSCIPLIBSHORTNAME).$(LIBEXT)

ALLSRC		+=	$(OBJSCIPLIBSRC)


#-----------------------------------------------------------------------------
# Main Program
#-----------------------------------------------------------------------------

MAINSHORTNAME	=	scip
MAINNAME	=	$(MAINSHORTNAME)-$(VERSION)

MAINOBJ		=	main.o
MAINSRC		=	$(addprefix $(SRCDIR)/,$(MAINOBJ:.o=.c))
MAINDEP		=	$(SRCDIR)/depend.main.$(OPT)

MAINFILE	=	$(BINDIR)/$(MAINNAME).$(BASE).$(LPS)$(EXEEXTENSION)
MAINOBJFILES	=	$(addprefix $(BINOBJDIR)/,$(MAINOBJ))
MAINLINK	=	$(BINDIR)/$(MAINSHORTNAME).$(BASE).$(LPS)$(EXEEXTENSION)
MAINSHORTLINK	=	$(BINDIR)/$(MAINSHORTNAME)$(EXEEXTENSION)
ALLSRC		+=	$(MAINSRC)

DLLFILENAME	=	lib$(MAINNAME).$(BASE).$(LPS).dll

LINKSMARKERFILE	=	$(LIBDIR)/$(LIBTYPE)/linkscreated.$(LPS)-$(LPSOPT).$(OSTYPE).$(ARCH).$(COMP)$(LINKLIBSUFFIX).$(ZIMPL)-$(ZIMPLOPT).$(IPOPT)-$(IPOPTOPT).$(GAMS)
LASTSETTINGS	=	$(OBJDIR)/make.lastsettings

#-----------------------------------------------------------------------------
# Rules
#-----------------------------------------------------------------------------

ifeq ($(VERBOSE),false)
.SILENT:	$(MAINFILE) $(SCIPLIBFILE) $(OBJSCIPLIBFILE) $(LPILIBFILE) $(NLPILIBFILE) \
		$(LPILIBLINK) $(LPILIBSHORTLINK) $(SCIPLIBLINK) $(SCIPLIBSHORTLINK) \
		$(OBJSCIPLIBLINK) $(OBJSCIPLIBSHORTLINK) $(NLPILIBLINK) $(NLPILIBSHORTLINK) \
		$(MAINLINK) $(MAINSHORTLINK) \
		$(LPILIBOBJFILES) $(NLPILIBOBJFILES) $(SCIPLIBOBJFILES) $(OBJSCIPLIBOBJFILES) $(MAINOBJFILES)
MAKE		+= -s
endif

.PHONY: all
all:		libs
		@$(MAKE) $(MAINFILE) $(MAINLINK) $(MAINSHORTLINK)

.PHONY: libs
libs:     	makesciplibfile
		@$(MAKE) $(OBJSCIPLIBFILE) $(LPILIBFILE) $(NLPILIBFILE) \
		$(LPILIBLINK) $(LPILIBSHORTLINK) $(NLPILIBLINK) $(NLPILIBSHORTLINK) \
		$(SCIPLIBLINK) $(SCIPLIBSHORTLINK) $(OBJSCIPLIBLINK) $(OBJSCIPLIBSHORTLINK)

.PHONY: preprocess
preprocess:     checkdefines
		@$(SHELL) -ec 'if test ! -e $(LINKSMARKERFILE) ; \
			then \
				echo "-> generating necessary links" ; \
				$(MAKE) -j1 $(LINKSMARKERFILE) ; \
			fi'
		@$(MAKE) touchexternal

.PHONY: lint
lint:		$(SCIPLIBSRC) $(OBJSCIPLIBSRC) $(MAINSRC)
		-rm -f lint.out
		@$(SHELL) -ec 'if test -e lint/co-gcc.mak ; \
			then \
				echo "-> generating gcc-include-path lint-file" ; \
				cd lint; $(MAKE) -f co-gcc.mak ; \
			else \
				echo "-> lint Makefile not found"; \
			fi'
ifeq ($(FILES),)
		$(SHELL) -ec 'for i in $^; \
			do \
				echo $$i; \
				$(LINT) lint/main-gcc.lnt +os\(lint.out\) -u -zero \
				$(FLAGS) -I/usr/include -UNDEBUG -UWITH_READLINE -UROUNDING_FE -D_BSD_SOURCE $$i; \
			done'
else
		$(SHELL) -ec  'for i in $(FILES); \
			do \
				echo $$i; \
				$(LINT) lint/main-gcc.lnt +os\(lint.out\) -u -zero \
				$(FLAGS) -I/usr/include -UNDEBUG -UWITH_READLINE -UROUNDING_FE -D_BSD_SOURCE $$i; \
			done'
endif

.PHONY: splint
splint:		$(SCIPLIBSRC) $(LPILIBSRC)
		-rm -f splint.out
ifeq ($(FILES),)
		$(SHELL) -c '$(SPLINT) -I$(SRCDIR) -I/usr/include/linux $(FLAGS) $(SPLINTFLAGS) $(filter %.c %.h,$^) >> splint.out;'
else
		$(SHELL) -c '$(SPLINT) -I$(SRCDIR) -I/usr/include/linux $(FLAGS) $(SPLINTFLAGS) $(filter %.c %.h,$(FILES)) >> splint.out;'
endif

.PHONY: doc
doc:
		cd doc; $(DOXY) $(MAINSHORTNAME).dxy; $(DOXY) $(MAINSHORTNAME)devel.dxy

.PHONY: docpreview
docpreview:
# generates preview for a list of files
ifneq ($(FILES),)
		echo "generating doxygen preview for $(FILES)"
		cd doc; ( cat $(MAINSHORTNAME).dxy && echo 'FILE_PATTERNS = $(FILES)' ) | $(DOXY) -
else
		echo "please specify file(s) for which preview should be created"
endif

.PHONY: check
check:		test

.PHONY: test
test:
		cd check; \
		$(SHELL) ./check.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) $(DISPFREQ) \
		$(CONTINUE) $(LOCK) $(VERSION) $(LPS) $(VALGRIND) $(CLIENTTMPDIR) $(REOPT) $(OPTCOMMAND) $(SETCUTOFF) $(MAXJOBS) $(VISUALIZE) $(PERMUTE) $(SEEDS);

.PHONY: testcount
testcount:
		cd check; \
		$(SHELL) ./check_count.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)).$(HOSTNAME) $(TIME) $(NODES) $(MEM) $(FEASTOL) \
		$(DISPFREQ) $(CONTINUE) $(LOCK) $(VERSION) $(LPS);

.PHONY: testcplex
testcplex:
		cd check; \
		$(SHELL) ./check.sh $(TEST) $(CPLEX) $(SETTINGS) $(notdir $(CPLEX)).$(OSTYPE).$(ARCH) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) $(DISPFREQ) \
		$(CONTINUE) $(LOCK) $(VERSION) $(LPS) $(VALGRIND) $(CLIENTTMPDIR) $(REOPT) $(OPTCOMMAND) $(SETCUTOFF) $(MAXJOBS) $(VISUALIZE);
.PHONY: testxpress
testxpress:
		cd check; \
		$(SHELL) ./check_xpress.sh $(TEST) $(XPRESS_BIN) $(SETTINGS) $(OSTYPE).$(ARCH).$(HOSTNAME) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) \
		$(DISPFREQ) $(CONTINUE);

.PHONY: testmosek
testmosek:
		cd check; \
		$(SHELL) ./check_mosek.sh $(TEST) $(MOSEK) $(SETTINGS) $(OSTYPE).$(ARCH).$(HOSTNAME) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) $(DISPFREQ) $(CONTINUE);

.PHONY: testcbc
testcbc:
		cd check; \
		$(SHELL) ./check_cbc.sh $(TEST) $(CBC) $(SETTINGS) $(OSTYPE).$(ARCH).$(HOSTNAME) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) $(CONTINUE);

.PHONY: testcbcparallel
testcbcparallel:
		cd check; \
                $(SHELL) ./check_cbc.sh $(TEST) $(CBCPARALLEL) $(SETTINGS) $(OSTYPE).$(ARCH).$(HOSTNAME) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) $(CONTINUE);

.PHONY: testgurobi
testgurobi:
		cd check; \
		$(SHELL) ./check_gurobi.sh $(TEST) $(GUROBI) $(SETTINGS) $(OSTYPE).$(ARCH).$(HOSTNAME) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) $(DISPFREQ) $(CONTINUE);

.PHONY: testglpk
testglpk:
		cd check; \
		$(SHELL) ./check_glpk.sh $(TEST) $(GLPK) $(SETTINGS) $(OSTYPE).$(ARCH).$(HOSTNAME) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) $(DISPFREQ) $(CONTINUE);

.PHONY: testsymphony
testsymphony:
		cd check; \
		$(SHELL) ./check_symphony.sh $(TEST) $(SYMPHONY) $(SETTINGS) $(OSTYPE).$(ARCH).$(HOSTNAME) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) \
		$(DISPFREQ) $(CONTINUE);

.PHONY: testblis
testblis:
		cd check; \
		$(SHELL) ./check_blis.sh $(TEST) $(BLIS) $(SETTINGS) $(OSTYPE).$(ARCH).$(HOSTNAME) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) $(DISPFREQ) $(CONTINUE);

.PHONY: tags
tags:
		rm -f TAGS; ctags -e -R -h ".c.cpp.h" --exclude=".*" src/; sed 's!\#undef .*!!g' TAGS > tags; mv tags TAGS

# include target to detect the current git hash
-include make/local/make.detectgithash

# this empty target is needed for the SCIP release versions
githash::      # do not remove the double-colon

# include local targets
-include make/local/make.targets

# include install/uninstall targets
-include make/make.install

# the testgams target need to come after make/local/make.targets has been included (if any), because the latter may assign a value to CLIENTTMPDIR
# if calling with GAMS=true, assume user wants the GAMS system that is linked in lib/gams.*
# if calling with GAMS=false (default), assume user has a GAMS system in the path (keeping original default behavior)
ifeq ($(GAMS),true)
	TESTGAMS = $(abspath $(GAMSDIR))/gams
else
ifeq ($(GAMS),false)
	TESTGAMS = gams
else
	TESTGAMS = $(GAMS)
endif
endif
.PHONY: testgams
testgams:
		cd check; \
		$(SHELL) ./check_gamscluster.sh $(TEST) $(TESTGAMS) "$(GAMSSOLVER)" $(SETTINGS) $(OSTYPE).$(ARCH) $(TIME) $(NODES) $(MEM) "$(GAP)" \
		$(THREADS) $(CONTINUE) "$(CONVERTSCIP)" local dummy dummy "$(CLIENTTMPDIR)" 1 true $(SETCUTOFF);

$(LPILIBLINK):	$(LPILIBFILE)
		@rm -f $@
		cd $(dir $@) && $(LN_s) $(notdir $(LPILIBFILE)) $(notdir $@)

# the short link targets should be phony such that they are always updated and point to the files with last make options, even if nothing needed to be rebuilt
.PHONY: $(LPILIBSHORTLINK)
$(LPILIBSHORTLINK):	$(LPILIBFILE)
		@rm -f $@
		cd $(dir $@) && $(LN_s) $(notdir $(LPILIBFILE)) $(notdir $@)

$(NLPILIBLINK):	$(NLPILIBFILE)
		@rm -f $@
		cd $(dir $@) && $(LN_s) $(notdir $(NLPILIBFILE)) $(notdir $@)

# the short link targets should be phony such that they are always updated and point to the files with last make options, even if nothing needed to be rebuilt
.PHONY: $(NLPILIBSHORTLINK)
$(NLPILIBSHORTLINK):	$(NLPILIBFILE)
		@rm -f $@
		cd $(dir $@) && $(LN_s) $(notdir $(NLPILIBFILE)) $(notdir $@)

$(SCIPLIBLINK):	$(SCIPLIBFILE)
		@rm -f $@
		cd $(dir $@) && $(LN_s) $(notdir $(SCIPLIBFILE)) $(notdir $@)

# the short link targets should be phony such that they are always updated and point to the files with last make options, even if nothing needed to be rebuilt
.PHONY: $(SCIPLIBSHORTLINK)
$(SCIPLIBSHORTLINK):	$(SCIPLIBFILE)
		@rm -f $@
		cd $(dir $@) && $(LN_s) $(notdir $(SCIPLIBFILE)) $(notdir $@)

$(OBJSCIPLIBLINK):	$(OBJSCIPLIBFILE)
		@rm -f $@
		cd $(dir $@) && $(LN_s) $(notdir $(OBJSCIPLIBFILE)) $(notdir $@)

# the short link targets should be phony such that they are always updated and point to the files with last make options, even if nothing needed to be rebuilt
.PHONY: $(OBJSCIPLIBSHORTLINK)
$(OBJSCIPLIBSHORTLINK):	$(OBJSCIPLIBFILE)
		@rm -f $@
		cd $(dir $@) && $(LN_s) $(notdir $(OBJSCIPLIBFILE)) $(notdir $@)

# the short link targets should be phony such that they are always updated and point to the files with last make options, even if nothing needed to be rebuilt
.PHONY: $(MAINSHORTLINK)
$(MAINLINK) $(MAINSHORTLINK):	$(MAINFILE)
		@rm -f $@
		cd $(dir $@) && $(LN_s) $(notdir $(MAINFILE)) $(notdir $@)

$(OBJDIR):
		@-mkdir -p $(OBJDIR)

$(BINOBJDIR):	| $(OBJDIR)
		@-mkdir -p $(BINOBJDIR)

$(LIBOBJDIR):	| $(OBJDIR)
		@-mkdir -p $(LIBOBJDIR)

$(LIBOBJSUBDIRS):	| $(LIBOBJDIR)
		@-mkdir -p $(LIBOBJSUBDIRS)

$(LIBDIR):
		@-mkdir -p $(LIBDIR)

$(LIBDIR)/static: $(LIBDIR)
		@-mkdir -p $(LIBDIR)/static

$(LIBDIR)/shared: $(LIBDIR)
		@-mkdir -p $(LIBDIR)/shared

$(LIBDIR)/include: $(LIBDIR)
		@-mkdir -p $(LIBDIR)/include

$(BINDIR):
		@-mkdir -p $(BINDIR)

.PHONY: clean
clean:          cleanlibs cleanbin | $(LIBOBJSUBDIRS) $(LIBOBJDIR) $(BINOBJDIR) $(OBJDIR)
ifneq ($(LIBOBJDIR),)
		@-(cd $(LIBOBJDIR) && rm -f */*.o)
		@-rmdir $(LIBOBJSUBDIRS)
		@-rmdir $(LIBOBJDIR)
endif
ifneq ($(BINOBJDIR),)
		@-rm -f $(BINOBJDIR)/*.o && rmdir $(BINOBJDIR)
endif
ifneq ($(OBJDIR),)
		@-rm -f $(LASTSETTINGS)
		@-rmdir $(OBJDIR)
endif

.PHONY: cleanlibs
cleanlibs:      | $(LIBDIR)/$(LIBTYPE)
		@echo "-> remove library $(SCIPLIBFILE)"
		@-rm -f $(SCIPLIBFILE) $(SCIPLIBLINK) $(SCIPLIBSHORTLINK)
		@echo "-> remove library $(OBJSCIPLIBFILE)"
		@-rm -f $(OBJSCIPLIBFILE) $(OBJSCIPLIBLINK) $(OBJSCIPLIBSHORTLINK)
		@echo "-> remove library $(LPILIBFILE)"
		@-rm -f $(LPILIBFILE) $(LPILIBLINK) $(LPILIBSHORTLINK)
		@echo "-> remove library $(NLPILIBFILE)"
		@-rm -f $(NLPILIBFILE) $(NLPILIBLINK) $(NLPILIBSHORTLINK)

.PHONY: cleanbin
cleanbin:       | $(BINDIR)
		@echo "-> remove binary $(MAINFILE)"
		@-rm -f $(MAINFILE) $(MAINLINK) $(MAINSHORTLINK)

.PHONY: lpidepend
lpidepend:
ifeq ($(LINKER),C)
		$(SHELL) -ec '$(DCC) $(FLAGS) $(DFLAGS) $(LPILIBSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o *: *$(SRCDIR)/\([0-9A-Za-z_/]*\).c|$$\(LIBOBJDIR\)/\2.o: $(SRCDIR)/\2.c|g'\'' \
		| sed '\''s|$(LIBDIR)/include/cpxinc/cpxconst.h||g'\'' \
		>$(LPILIBDEP)'
endif
ifeq ($(LINKER),CPP)
		$(SHELL) -ec '$(DCXX) $(FLAGS) $(DFLAGS) $(LPILIBSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o *: *$(SRCDIR)/\([0-9A-Za-z_/]*\).c|$$\(LIBOBJDIR\)/\2.o: $(SRCDIR)/\2.c|g'\'' \
		| sed '\''s|$(LIBDIR)/clp[^ ]*||g'\'' \
		| sed '\''s|$(LIBDIR)/include/cpxinc/cpxconst.h||g'\'' \
		| sed '\''s|$(LIBDIR)/include/spxinc[^ ]*||g'\'' \
		>$(LPILIBDEP)'
endif
# We explicitely add all lpi's here, since the content of depend.lpscheck should be independent of the currently selected LPI,
# but contain all LPI's that use the WITH_LPSCHECK define.
		@echo `grep -l "WITH_LPSCHECK" $(SCIPLIBSRC) $(OBJSCIPLIBSRC) $(MAINSRC) $(NLPILIBSRC) src/lpi/lpi*.{c,cpp}` >$(LPSCHECKDEP)

.PHONY: nlpidepend
nlpidepend:
ifeq ($(LINKER),C)
		$(SHELL) -ec '$(DCC) $(FLAGS) $(DFLAGS) $(NLPILIBSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o *: *$(SRCDIR)/\([0-9A-Za-z_/]*\).c|$$\(LIBOBJDIR\)/\2.o: $(SRCDIR)/\2.c|g'\'' \
		>$(NLPILIBDEP)'
endif
ifeq ($(LINKER),CPP)
		$(SHELL) -ec '$(DCXX) $(FLAGS) $(DFLAGS) $(NLPILIBSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o *: *$(SRCDIR)/\([0-9A-Za-z_/]*\).c|$$\(LIBOBJDIR\)/\2.o: $(SRCDIR)/\2.c|g'\'' \
		| sed '\''s|$(LIBDIR)/ipopt[^ ]*||g'\'' \
		>$(NLPILIBDEP)'
endif

.PHONY: maindepend
maindepend:
		$(SHELL) -ec '$(DCC) $(FLAGS) $(DFLAGS) $(MAINSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o *: *$(SRCDIR)/\([0-9A-Za-z_/]*\).c|$$\(BINOBJDIR\)/\2.o: $(SRCDIR)/\2.c|g'\'' \
		>$(MAINDEP)'

.PHONY: objscipdepend
objscipdepend:
		$(SHELL) -ec '$(DCXX) $(FLAGS) $(DFLAGS) $(OBJSCIPLIBSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o *: *$(SRCDIR)/\([0-9A-Za-z_/]*\).c|$$\(LIBOBJDIR\)/\2.o: $(SRCDIR)/\2.c|g'\'' \
		>$(OBJSCIPLIBDEP)'

.PHONY: scipdepend
scipdepend:
		$(SHELL) -ec '$(DCC) $(FLAGS) $(DFLAGS) $(SCIPLIBSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o *: *$(SRCDIR)/\([0-9A-Za-z_/]*\).c|$$\(LIBOBJDIR\)/\2.o: $(SRCDIR)/\2.c|g'\'' \
		>$(SCIPLIBDEP)'
		@echo `grep -l "WITH_ZLIB" $(ALLSRC)` >$(ZLIBDEP)
		@echo `grep -l "WITH_GMP" $(ALLSRC)` >$(GMPDEP)
		@echo `grep -l "WITH_READLINE" $(ALLSRC)` >$(READLINEDEP)
		@echo `grep -l "WITH_ZIMPL" $(ALLSRC)` >$(ZIMPLDEP)
		@echo `grep -l "WITH_GAMS" $(ALLSRC)` >$(GAMSDEP)
		@echo `grep -l "NPARASCIP" $(ALLSRC)` >$(PARASCIPDEP)

depend:		scipdepend lpidepend nlpidepend maindepend objscipdepend

-include	$(MAINDEP)
-include	$(SCIPLIBDEP)
-include	$(OBJSCIPLIBDEP)
-include	$(LPILIBDEP)
-include	$(NLPILIBDEP)

$(MAINFILE):	$(MAINOBJFILES) $(SCIPLIBFILE) | $(BINDIR) $(BINOBJDIR) $(LIBOBJSUBDIRS)
		@echo "-> linking $@"
ifeq ($(LINKER),C)
		-$(LINKCC) $(MAINOBJFILES) $(LINKCCSCIPALL) $(LDFLAGS) $(LINKCC_o)$@
		|| ($(MAKE) errorhints && false)
endif
ifeq ($(LINKER),CPP)
		-$(LINKCXX) $(MAINOBJFILES) $(LINKCCSCIPALL) $(LDFLAGS) $(LINKCXX_o)$@ \
		|| ($(MAKE) errorhints && false)
endif


.PHONY: makesciplibfile
makesciplibfile: preprocess
		@$(MAKE) $(SCIPLIBFILE)

$(SCIPLIBFILE):	$(SCIPLIBOBJFILES) | $(LIBDIR)/$(LIBTYPE) $(LIBOBJSUBDIRS)
		@echo "-> generating library $@"
		-rm -f $@
		$(LIBBUILD) $(LIBBUILDFLAGS) $(LIBBUILD_o)$@ $(SCIPLIBOBJFILES) $(SCIPLIBEXTLIBS)
ifneq ($(RANLIB),)
		$(RANLIB) $@
endif

.PHONY: makeobjsciplibfile
$(OBJSCIPLIBFILE):	$(OBJSCIPLIBOBJFILES) | $(LIBOBJSUBDIRS) $(LIBDIR)/$(LIBTYPE)
		@echo "-> generating library $@"
		-rm -f $@
		$(LIBBUILD) $(LIBBUILDFLAGS) $(LIBBUILD_o)$@ $(OBJSCIPLIBOBJFILES)
ifneq ($(RANLIB),)
		$(RANLIB) $@
endif

.PHONY: makelpilibfile
$(LPILIBFILE):	$(LPILIBOBJFILES) | $(LIBOBJSUBDIRS) $(LIBDIR)/$(LIBTYPE)
		@echo "-> generating library $@"
		-rm -f $@
		$(LIBBUILD) $(LIBBUILDFLAGS) $(LIBBUILD_o)$@ $(LPILIBOBJFILES) $(LPILIBEXTLIBS)
ifneq ($(RANLIB),)
		$(RANLIB) $@
endif

.PHONY: makenlpilibfile
$(NLPILIBFILE):	$(NLPILIBOBJFILES) $(NLPILIBSCIPOBJFILES) | $(LIBOBJSUBDIRS) $(LIBDIR)/$(LIBTYPE)
		@echo "-> generating library $@"
		-rm -f $@
		$(LIBBUILD) $(LIBBUILDFLAGS) $(LIBBUILD_o)$@ $(NLPILIBOBJFILES) $(NLPILIBSCIPOBJFILES) $(NLPILIBEXTLIBS)
ifneq ($(RANLIB),)
		$(RANLIB) $@
endif

$(BINOBJDIR)/%.o:	$(SRCDIR)/%.c | $(BINOBJDIR)
		@echo "-> compiling $@"
		$(CC) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CFLAGS) $(CC_c)$< $(CC_o)$@

$(BINOBJDIR)/%.o:	$(SRCDIR)/%.cpp | $(BINOBJDIR)
		@echo "-> compiling $@"
		$(CXX) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CXXFLAGS) $(CXX_c)$< $(CXX_o)$@

$(LIBOBJDIR)/%.o:	$(SRCDIR)/%.c | $(LIBOBJDIR) $(LIBOBJSUBDIRS)
		@echo "-> compiling $@"
		$(CC) $(FLAGS) $(OFLAGS) $(LIBOFLAGS) $(CFLAGS) $(CC_c)$< $(CC_o)$@

$(LIBOBJDIR)/%.o:	$(SRCDIR)/%.cpp | $(LIBOBJDIR) $(LIBOBJSUBDIRS)
		@echo "-> compiling $@"
		$(CXX) $(FLAGS) $(OFLAGS) $(LIBOFLAGS) $(CXXFLAGS) $(CXX_c)$< $(CXX_o)$@

ifeq ($(GAMS),true)
$(LIBOBJDIR)/scip/%.o:	$(GAMSDIR)/apifiles/C/api/%.c | $(LIBOBJDIR)
		@echo "-> compiling $@"
		$(CC) $(FLAGS) $(OFLAGS) $(LIBOFLAGS) $(CFLAGS) $(CC_c)$< $(CC_o)$@
$(LIBOBJDIR)/scip/%.o:	$(SRCDIR)/../interfaces/gams/src/%.c | $(LIBOBJDIR)
		@echo "-> compiling $@"
		$(CC) $(FLAGS) $(OFLAGS) $(LIBOFLAGS) $(CFLAGS) $(CC_c)$< $(CC_o)$@
endif

-include $(LASTSETTINGS)

.PHONY: dll
dll: $(SCIPLIBOBJFILES) $(MAINOBJFILES) $(LPILIBOBJFILES) $(NLPILIBOBJFILES) $(OBJSCIPLIBOBJFILES) | $(LIBOBJSUBDIRS) $(LIBDIR)/$(LIBTYPE)
		@echo "-> generating library $@"
		$(LINKCC) $(LIBBUILDFLAGS) $(LINKCC_L)$(LIBDIR)/$(LIBTYPE) $(LIBBUILD_o)$(LIBDIR)/$(LIBTYPE)/$(DLLFILENAME) \
			$(SCIPLIBOBJFILES) $(OBJSCIPLIBOBJFILES) $(NLPILIBOBJFILES) $(LPILIBOBJFILES) \
			$(LPSLDFLAGS) $(LDFLAGS)

.PHONY: touchexternal
touchexternal:	$(ZLIBDEP) $(GMPDEP) $(READLINEDEP) $(ZIMPLDEP) $(GAMSDEP) $(LPSCHECKDEP) $(PARASCIPDEP) | $(LIBOBJDIR)
ifeq ($(TOUCHLINKS),true)
		@-touch $(ZLIBSRC)
		@-touch $(GMPSRC)
		@-touch $(READLINESRC)
		@-touch $(ZIMPLSRC)
		@-touch $(GAMSSRC)
		@-touch $(LPSCHECKSRC)
		@-touch $(LPILIBSRC)
		@-touch $(NLPILIBSRC)
endif
ifneq ($(SCIPGITHASH),$(LAST_SCIPGITHASH))
		@$(MAKE) githash
endif
		@$(SHELL) -ec 'if test ! -e $(SCIPGITHASHFILE) ; \
			then \
				echo "-> generating $(SCIPGITHASHFILE)" ; \
				$(MAKE) githash ; \
			fi'
ifneq ($(subst \\n,\n,$(BUILDFLAGS)),$(LAST_BUILDFLAGS))
		@echo "#define SCIP_BUILDFLAGS \"$(BUILDFLAGS)\"" > $(SCIPBUILDFLAGSFILE)
endif
ifneq ($(ZLIB),$(LAST_ZLIB))
		@-touch $(ZLIBSRC)
endif
ifneq ($(GMP),$(LAST_GMP))
		@-touch $(GMPSRC)
endif
ifneq ($(READLINE),$(LAST_READLINE))
		@-touch $(READLINESRC)
endif
ifneq ($(ZIMPL),$(LAST_ZIMPL))
		@-touch $(ZIMPLSRC)
endif
ifneq ($(GAMS),$(LAST_GAMS))
		@-touch $(GAMSSRC)
endif
ifneq ($(LPSCHECK),$(LAST_LPSCHECK))
		@-touch $(LPSCHECKSRC)
endif
ifneq ($(PARASCIP),$(LAST_PARASCIP))
		@-touch $(PARASCIPSRC)
endif
ifneq ($(USRFLAGS),$(LAST_USRFLAGS))
		@-touch $(ALLSRC)
endif
ifneq ($(USROFLAGS),$(LAST_USROFLAGS))
		@-touch $(ALLSRC)
endif
ifneq ($(USRCFLAGS),$(LAST_USRCFLAGS))
		@-touch $(ALLSRC)
endif
ifneq ($(USRCXXFLAGS),$(LAST_USRCXXFLAGS))
		@-touch $(ALLSRC)
endif
ifneq ($(USRLDFLAGS),$(LAST_USRLDFLAGS))
		@-touch -c $(SCIPLIBOBJFILES) $(LPILIBOBJFILES) $(NLPILIBOBJFILES) $(MAINOBJFILES)
endif
ifneq ($(USRARFLAGS),$(LAST_USRARFLAGS))
		@-touch -c $(SCIPLIBOBJFILES) $(OBJSCIPLIBOBJFILES) $(LPILIBOBJFILES) $(NLPILIBOBJFILES) $(NLPILIBSCIPOBJFILES)
endif
ifneq ($(NOBLKMEM),$(LAST_NOBLKMEM))
		@-touch -c $(ALLSRC)
endif
ifneq ($(NOBUFMEM),$(LAST_NOBUFMEM))
		@-touch -c $(ALLSRC)
endif
ifneq ($(NOBLKBUFMEM),$(LAST_NOBLKBUFMEM))
		@-touch -c $(ALLSRC)
endif
ifneq ($(SANITIZE),$(LAST_SANITIZE))
		@-touch -c $(ALLSRC)
endif
		@-rm -f $(LASTSETTINGS)
		@echo "LAST_BUILDFLAGS=\"$(BUILDFLAGS)\"" >> $(LASTSETTINGS)
		@echo "LAST_SCIPGITHASH=$(SCIPGITHASH)" >> $(LASTSETTINGS)
		@echo "LAST_ZLIB=$(ZLIB)" >> $(LASTSETTINGS)
		@echo "LAST_GMP=$(GMP)" >> $(LASTSETTINGS)
		@echo "LAST_READLINE=$(READLINE)" >> $(LASTSETTINGS)
		@echo "LAST_ZIMPL=$(ZIMPL)" >> $(LASTSETTINGS)
		@echo "LAST_GAMS=$(GAMS)" >> $(LASTSETTINGS)
		@echo "LAST_PARASCIP=$(PARASCIP)" >> $(LASTSETTINGS)
		@echo "LAST_LPSCHECK=$(LPSCHECK)" >> $(LASTSETTINGS)
		@echo "LAST_USRFLAGS=$(USRFLAGS)" >> $(LASTSETTINGS)
		@echo "LAST_USROFLAGS=$(USROFLAGS)" >> $(LASTSETTINGS)
		@echo "LAST_USRCFLAGS=$(USRCFLAGS)" >> $(LASTSETTINGS)
		@echo "LAST_USRCXXFLAGS=$(USRCXXFLAGS)" >> $(LASTSETTINGS)
		@echo "LAST_USRLDFLAGS=$(USRLDFLAGS)" >> $(LASTSETTINGS)
		@echo "LAST_USRARFLAGS=$(USRARFLAGS)" >> $(LASTSETTINGS)
		@echo "LAST_USRDFLAGS=$(USRDFLAGS)" >> $(LASTSETTINGS)
		@echo "LAST_NOBLKMEM=$(NOBLKMEM)" >> $(LASTSETTINGS)
		@echo "LAST_NOBUFMEM=$(NOBUFMEM)" >> $(LASTSETTINGS)
		@echo "LAST_NOBLKBUFMEM=$(NOBLKBUFMEM)" >> $(LASTSETTINGS)
		@echo "LAST_SANITIZE=$(SANITIZE)" >> $(LASTSETTINGS)

$(LINKSMARKERFILE):
		@$(MAKE) links

.PHONY: links
links:		| $(LIBDIR)/static $(LIBDIR)/shared $(DIRECTORIES) echosoftlinks $(SOFTLINKS)
		@rm -f $(LINKSMARKERFILE)
		@echo "this is only a marker" > $(LINKSMARKERFILE)

.PHONY: echosoftlinks
echosoftlinks:
		@echo
		@echo "- Current settings: LPS=$(LPS) OSTYPE=$(OSTYPE) ARCH=$(ARCH) COMP=$(COMP) SHARED=$(SHARED) SUFFIX=$(LINKLIBSUFFIX) ZIMPL=$(ZIMPL) ZIMPLOPT=$(ZIMPLOPT) IPOPT=$(IPOPT) IPOPTOPT=$(IPOPTOPT) EXPRINT=$(EXPRINT) GAMS=$(GAMS)"
		@echo
		@echo "* SCIP needs some softlinks to external programs, in particular, LP-solvers."
		@echo "* Please insert the paths to the corresponding directories/libraries below."
		@echo "* The links will be installed in the 'lib/include' and 'lib/$(LIBTYPE)' directories."
		@echo "* For more information and if you experience problems see the INSTALL file."
		@echo
		@echo -e $(LPIINSTMSG)

$(DIRECTORIES):
		@echo
		@echo "- creating directory \"$@\""
		@-mkdir -p $@


# Create softlinks for external libraries. The user can enter the
# filename and the link is created in the corresponding directories
# (lib/include, lib/shared, lib/static).
#
# We also test whether older links in lib/ still exist. In later
# releases this feature can be removed.
.PHONY: $(SOFTLINKS)
$(SOFTLINKS):
ifeq ($(MAKESOFTLINKS), true)
		@$(SHELL) -ec 'if test ! -e $@ ; \
			then \
				DIRNAME=`dirname $@` ; \
				echo ; \
				echo "> Enter soft-link target file or directory for \"$@\" (return if not needed): " ; \
				echo -n "> " ; \
				cd $$DIRNAME ; \
				eval $(READ) TARGET ; \
				cd $(SCIPREALPATH) ; \
				if test "$$TARGET" != "" ; \
				then \
					echo "-> creating softlink \"$@\" -> \"$$TARGET\"" ; \
					rm -f $@ ; \
					$(LN_s) $$TARGET $@ ; \
				else \
					echo "* skipped creation of softlink \"$@\". Call \"make links\" if needed later." ; \
				fi ; \
				FILENAME=$@ ; \
				FNAME=$${FILENAME#lib/static} ; \
				FNAME=$${FNAME#lib/include} ; \
				FNAME="lib"$$FNAME ; \
				if test -e $$FNAME ; \
				then \
					echo ; \
					echo "The link "$$FNAME" still exists. Consider removing it." ; \
				fi ; \
				echo ; \
			fi'
endif

.PHONY: checkdefines
checkdefines:
ifeq ($(LPILIBOBJ),)
		$(error invalid LP solver selected: LPS=$(LPS). Possible options are: $(LPSOPTIONS))
endif
ifneq ($(GMP),true)
ifneq ($(GMP),false)
		$(error invalid GMP flag selected: GMP=$(GMP). Possible options are: true false)
endif
endif
ifneq ($(ZIMPL),true)
ifneq ($(ZIMPL),false)
		$(error invalid ZIMPL flag selected: ZIMPL=$(ZIMPL). Possible options are: true false auto)
endif
endif
ifneq ($(GAMS),true)
ifneq ($(GAMS),false)
		$(error invalid GAMS flag selected: GAMS=$(GAMS). Possible options are: true false)
endif
endif
ifneq ($(IPOPT),true)
ifneq ($(IPOPT),false)
		$(error invalid IPOPT flag selected: IPOPT=$(IPOPT). Possible options are: true false)
endif
endif
ifneq ($(READLINE),true)
ifneq ($(READLINE),false)
		$(error invalid READLINE flag selected: READLINE=$(READLINE). Possible options are: true false)
endif
endif
ifneq ($(ZLIB),true)
ifneq ($(ZLIB),false)
		$(error invalid ZLIB flag selected: ZLIB=$(ZLIB). Possible options are: true false)
endif
endif
ifneq ($(PARASCIP),true)
ifneq ($(PARASCIP),false)
		$(error invalid PARASCIP flag selected: PARASCIP=$(PARASCIP). Possible options are: true false)
endif
endif
ifeq ($(SHARED),true)
ifeq ($(COMP),msvc)
		$(error invalid flags selected: SHARED=$(SHARED) and COMP=$(COMP). Please use 'make dll' to generate a dynamic library with MSVC)
endif
endif

.PHONY: errorhints
errorhints:
ifeq ($(READLINE),true)
		@echo "build failed with READLINE=true: if readline is not available, try building with READLINE=false"
endif
ifeq ($(ZLIB),true)
		@echo "build failed with ZLIB=true: if ZLIB is not available, try building with ZLIB=false"
endif
ifeq ($(GMP),true)
		@echo "build failed with GMP=true: if GMP is not available, try building with GMP=false (note that this will deactivate Zimpl support)"
endif
ifeq ($(GMP),false)
ifeq ($(LPS),spx1)
		@echo "build failed with GMP=false and LPS=spx1: use GMP=true or make sure that SoPlex is also built without GMP support (make GMP=false)"
endif
ifeq ($(LPS),spx2)
		@echo "build failed with GMP=false and LPS=spx2: use GMP=true or make sure that SoPlex is also built without GMP support (make GMP=false)"
endif
endif

.PHONY: help
help:
		@echo "Use the SCIP makefile system."
		@echo
		@echo "  The main options for the SCIP makefile system are as follows:"
		@echo
		@echo "  General commands:"
		@echo "  - OPT={dbg|opt}: Use debug or optimized (default) mode, respectively."
		@echo "  - LPS={clp|cpx|grb|msk|qso|spx|xprs|none}: Determine LP-solver."
		@echo "      clp: COIN-OR Clp LP-solver"
		@echo "      cpx: CPLEX LP-solver"
		@echo "      grb: Gurobi LP-solver (interface is in beta stage)"
		@echo "      msk: Mosek LP-solver"
		@echo "      qso: QSopt LP-solver"
		@echo "      spx: old SoPlex LP-solver (for versions < 2)"
		@echo "      spx2: new SoPlex LP-solver (default) (from version 2)"
		@echo "      xprs: XPress LP-solver"
		@echo "      none: no LP-solver"
		@echo "  - COMP={clang|gnu|intel}: Determine compiler."
		@echo "  - SHARED={true|false}: Build shared libraries or not (default)."
		@echo
		@echo "  More detailed options:"
		@echo "  - ZIMPL=<true|false>: Turn ZIMPL support on (default) or off."
		@echo "  - ZIMPLOPT=<dbg|opt>: Use debug or optimized (default) mode for ZIMPL."
		@echo "  - LPSOPT=<dbg|opt>: Use debug or optimized (default) mode for LP-solver (SoPlex and Clp only)."
		@echo "  - READLINE=<true|false>: Turns support via the readline library on (default) or off."
		@echo "  - IPOPT=<true|false>: Turns support of IPOPT on or off (default)."
		@echo "  - EXPRINT=<cppad|none>: Use CppAD as expressions interpreter (default) or no expressions interpreter."
		@echo "  - GAMS=<true|false>: To enable or disable (default) reading functionality in GAMS reader (needs GAMS)."
		@echo "  - NOBLKMEM=<true|false>: Turn off block memory or on (default)."
		@echo "  - NOBUFMEM=<true|false>>: Turn off buffer memory or on (default)."
		@echo "  - NOBLKBUFMEM=<true|false>: Turn usage of internal memory functions off or on (default)."
		@echo "  - VERBOSE=<true|false>: Turn on verbose messages of makefile or off (default)."
		@echo
		@echo "  The main targets are:"
		@echo "  - all (default): Build SCIP libary and binary."
		@echo "  - makesciplibfile: Make libscip for the main part of SCIP."
		@echo "  - makeobjsciplibfile: Make libscipobjs for the C++-interface of SCIP."
		@echo "  - makelpilibfile: Make liblpi for the LP interface in SCIP."
		@echo "  - makenlpilibfile: Make libnlpi for the NLP interface in SCIP."
		@echo "  - links: Reconfigures the links in the \"lib\" directory."
		@echo "  - doc: Creates documentation in the \"doc\" directory."
		@echo "  - libs: Create all SCIP libaries."
		@echo "  - lint: Run lint on all SCIP files. (Need flexelint.)"
		@echo "  - lint: Run splint on all C SCIP files. (Need splint.)"
		@echo "  - clean: Removes all object files."
		@echo "  - cleanlibs: Remove all SCIP libraries."
		@echo "  - depend: Creates dependencies files. This is only needed if you add files to SCIP."
		@echo "  - tags: Creates TAGS file that can be used in (x)emacs."
		@echo "  - check or test: Runs the check/test script, see the online documentation."

# --- EOF ---------------------------------------------------------------------
# DO NOT DELETE
