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
#pragma ident "@(#) $Id: heur_rounding.h,v 1.13 2006/01/03 12:22:48 bzfpfend Exp $"

/**@file   heur_rounding.h
 * @brief  LP rounding heuristic that tries to recover from intermediate infeasibilities
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_HEUR_ROUNDING_H__
#define __SCIP_HEUR_ROUNDING_H__


#include "scip/scip.h"


/** creates the rounding heuristic with infeasibility recovering and includes it in SCIP */
extern
SCIP_RETCODE SCIPincludeHeurRounding(
   SCIP*                 scip                /**< SCIP data structure */
   );

#endif
