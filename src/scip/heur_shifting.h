/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2008 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: heur_shifting.h,v 1.3 2008/04/17 17:49:09 bzfpfets Exp $"

/**@file   heur_shifting.h
 * @brief  LP rounding heuristic that tries to recover from intermediate infeasibilities and shifts continuous variables
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_HEUR_SHIFTING_H__
#define __SCIP_HEUR_SHIFTING_H__


#include "scip/scip.h"


/** creates the shifting heuristic and includes it in SCIP */
extern
SCIP_RETCODE SCIPincludeHeurShifting(
   SCIP*                 scip                /**< SCIP data structure */
   );

#endif
