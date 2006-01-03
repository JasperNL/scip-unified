/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2006 Tobias Achterberg                              */
/*                                                                           */
/*                  2002-2006 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: probdata_healthcare.h,v 1.2 2006/01/03 12:22:39 bzfpfend Exp $"

#ifndef __HCP_PROBDATA_HEALTHCARE__
#define __HCP_PROBDATA_HEALTHCARE__

#include "scip/scip.h"


extern
SCIP_RETCODE HCPcreateProbHealthcare(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< problem name */
   );

extern
SCIP_RETCODE HCPgenerateModel(
   SCIP*                 scip                /**< SCIP data structure */
   );

extern
SCIP_CONS** HCPgetConsServejobs(
   SCIP*                 scip                /**< SCIP data structure */
   );

extern
SCIP_CONS** HCPgetConsWorkers(
   SCIP*                 scip                /**< SCIP data structure */
   );

extern
int HCPgetNJobs(
   SCIP*                 scip                /**< SCIP data structure */
   );

extern
int HCPgetNWorkers(
   SCIP*                 scip                /**< SCIP data structure */
   );


#endif
