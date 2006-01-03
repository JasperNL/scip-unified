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
#pragma ident "@(#) $Id: cons_conjunction.h,v 1.10 2006/01/03 12:22:44 bzfpfend Exp $"

/**@file   cons_conjunction.h
 * @brief  constraint handler for conjunction constraints
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_CONS_CONJUNCTION_H__
#define __SCIP_CONS_CONJUNCTION_H__


#include "scip/scip.h"


/** creates the handler for conjunction constraints and includes it in SCIP */
extern
SCIP_RETCODE SCIPincludeConshdlrConjunction(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates and captures a conjunction constraint */
extern
SCIP_RETCODE SCIPcreateConsConjunction(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   int                   nconss,             /**< number of initial constraints in conjunction */
   SCIP_CONS**           conss,              /**< initial constraint in conjunction */
   SCIP_Bool             enforce,            /**< should the constraint be enforced during node processing? */
   SCIP_Bool             check,              /**< should the constraint be checked for feasibility? */
   SCIP_Bool             local,              /**< is constraint only valid locally? */
   SCIP_Bool             modifiable,         /**< is constraint modifiable (subject to column generation)? */
   SCIP_Bool             dynamic             /**< is constraint subject to aging? */
   );

/** adds constraint to the conjunction of constraints */
extern
SCIP_RETCODE SCIPaddConsElemConjunction(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< conjunction constraint */
   SCIP_CONS*            addcons             /**< additional constraint in conjunction */
   );

#endif
