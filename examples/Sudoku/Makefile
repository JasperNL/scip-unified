#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2002-2020 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic License.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#@file    Makefile
#@brief   Makefile for Sudoku example using SCIP
#@author  Naga Venkata Chaitanya Gudapati


#-----------------------------------------------------------------------------
# path
#-----------------------------------------------------------------------------

SCIPDIR         =       ../..


#-----------------------------------------------------------------------------
# include default project Makefile from SCIP (need to do this twice, once to
# find the correct binary, then, after getting the correct flags from the
# binary (which is necessary since the ZIMPL flags differ from the default
# if compiled with the SCIP Optsuite instead of SCIP), we need to set the
# compile flags, e.g., for the ZIMPL library, which is again done in make.project
#-----------------------------------------------------------------------------
include $(SCIPDIR)/make/make.project
SCIPVERSION				:=$(shell $(SCIPDIR)/bin/scip.$(BASE).$(LPS).$(TPI)$(EXEEXTENSION) -v | sed -e 's/$$/@/')
override ARCH			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* ARCH=\([^@]*\).*/\1/')
override EXPRINT		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* EXPRINT=\([^@]*\).*/\1/')
override GAMS			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* GAMS=\([^@]*\).*/\1/')
override GMP			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* GMP=\([^@]*\).*/\1/')
override SYM			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* SYM=\([^@]*\).*/\1/')
override IPOPT			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* IPOPT=\([^@]*\).*/\1/')
override IPOPTOPT		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* IPOPTOPT=\([^@]*\).*/\1/')
override LPSCHECK		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* LPSCHECK=\([^@]*\).*/\1/')
override LPSOPT 		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* LPSOPT=\([^@]*\).*/\1/')
override NOBLKBUFMEM	:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* NOBLKBUFMEM=\([^@]*\).*/\1/')
override NOBLKMEM		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* NOBLKMEM=\([^@]*\).*/\1/')
override NOBUFMEM		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* NOBUFMEM=\([^@]*\).*/\1/')
override PARASCIP		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* PARASCIP=\([^@]*\).*/\1/')
override READLINE		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* READLINE=\([^@]*\).*/\1/')
override SANITIZE		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* SANITIZE=\([^@]*\).*/\1/')
override ZIMPL			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* ZIMPL=\([^@]*\).*/\1/')
override ZIMPLOPT		:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* ZIMPLOPT=\([^@]*\).*/\1/')
override ZLIB			:=$(shell echo "$(SCIPVERSION)" | sed -e 's/.* ZLIB=\([^@]*\).*/\1/')
include $(SCIPDIR)/make/make.project

#-----------------------------------------------------------------------------
# Main Program
#-----------------------------------------------------------------------------

MAINNAME		=	sudoku
MAINOBJ			=	sudoku_main.o
MAINSRC			=	$(addprefix $(SRCDIR)/,$(MAINOBJ:.o=.cpp))


MAIN			=	$(MAINNAME).$(BASE).$(LPS)$(EXEEXTENSION)
MAINFILE		=	$(BINDIR)/$(MAIN)
MAINSHORTLINK	=	$(BINDIR)/$(MAINNAME)
MAINOBJFILES	=	$(addprefix $(OBJDIR)/,$(MAINOBJ))


#-----------------------------------------------------------------------------
# Rules
#-----------------------------------------------------------------------------

ifeq ($(VERBOSE),false)
.SILENT:	$(MAINFILE) $(MAINOBJFILES) $(MAINSHORTLINK)
endif

.PHONY: all
all:            $(SCIPDIR) $(MAINFILE) $(MAINSHORTLINK)

.PHONY: lint
lint:		$(MAINSRC)
		-rm -f lint.out
		$(SHELL) -ec 'for i in $^; \
			do \
			echo $$i; \
			$(LINT) -I$(SCIPDIR) lint/main-gcc.lnt +os\(lint.out\) -u -zero \
			$(FLAGS) -UNDEBUG -USCIP_WITH_READLINE -USCIP_ROUNDING_FE $$i; \
			done'

.PHONY: scip
scip:
		@$(MAKE) -C $(SCIPDIR) libs $^

$(MAINSHORTLINK):	$(MAINFILE)
		@rm -f $@
		cd $(dir $@) && ln -s $(notdir $(MAINFILE)) $(notdir $@)

$(OBJDIR):
		@-mkdir -p $(OBJDIR)

$(BINDIR):
		@-mkdir -p $(BINDIR)

.PHONY: clean
clean:		$(OBJDIR)
ifneq ($(OBJDIR),)
		-rm -f $(OBJDIR)/*.o
		-rmdir $(OBJDIR)
endif
		-rm -f $(MAINFILE)
		-rm -f $(BINDIR)/$(MAINNAME)

-include	$(MAINOBJFILES:.o=.d)

# main target
$(MAINFILE):	$(BINDIR) $(OBJDIR) $(SCIPLIBFILE) $(LPILIBFILE) $(NLPILIBFILE) $(MAINOBJFILES)
		@echo "-> linking $@"
		$(LINKCXX) $(MAINOBJFILES) $(LINKCXXSCIPALL) $(LDFLAGS) $(LINKCXX_o)$@

$(OBJDIR)/%.o:	$(SRCDIR)/%.cpp $(SRCDIR)/sudoku_utils.h
		@echo "-> compiling $@"
		$(CXX) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CXXFLAGS) -c $< $(CXX_o)$@



#---- EOF --------------------------------------------------------------------