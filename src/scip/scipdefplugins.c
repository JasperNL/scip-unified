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
#pragma ident "@(#) $Id: scipdefplugins.c,v 1.73 2008/04/17 17:49:17 bzfpfets Exp $"

/**@file   scipdefplugins.c
 * @brief  default SCIP plugins
 * @author Tobias Achterberg
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scip/scipdefplugins.h"
#include "scip/debug.h"



/** includes default SCIP plugins into SCIP */
SCIP_RETCODE SCIPincludeDefaultPlugins(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( SCIPincludeConshdlrLinear(scip) ); /* linear must be first due to constraint upgrading */
   SCIP_CALL( SCIPincludeConshdlrAnd(scip) );
   SCIP_CALL( SCIPincludeConshdlrBounddisjunction(scip) );
   SCIP_CALL( SCIPincludeConshdlrConjunction(scip) );
   SCIP_CALL( SCIPincludeConshdlrCountsols(scip) );
   SCIP_CALL( SCIPincludeConshdlrIntegral(scip) );
   SCIP_CALL( SCIPincludeConshdlrKnapsack(scip) );
   SCIP_CALL( SCIPincludeConshdlrLogicor(scip) );
   SCIP_CALL( SCIPincludeConshdlrOr(scip) );
   SCIP_CALL( SCIPincludeConshdlrSetppc(scip) );
   SCIP_CALL( SCIPincludeConshdlrSOS1(scip) );
   SCIP_CALL( SCIPincludeConshdlrSOS2(scip) );
   SCIP_CALL( SCIPincludeConshdlrVarbound(scip) );
   SCIP_CALL( SCIPincludeConshdlrXor(scip) );

#if 0
   SCIP_CALL( SCIPincludeConshdlrEqknapsack(scip) );
   SCIP_CALL( SCIPincludeConshdlrInvarknapsack(scip) );
   SCIP_CALL( SCIPincludeConshdlrBinpack(scip) );
#endif

   SCIP_CALL( SCIPincludeReaderCip(scip) );
   SCIP_CALL( SCIPincludeReaderCnf(scip) );
   SCIP_CALL( SCIPincludeReaderFix(scip) );
   SCIP_CALL( SCIPincludeReaderLp(scip) );
   SCIP_CALL( SCIPincludeReaderMps(scip) );
   SCIP_CALL( SCIPincludeReaderOpb(scip) );
   SCIP_CALL( SCIPincludeReaderPpm(scip) );
   SCIP_CALL( SCIPincludeReaderRlp(scip) );
   SCIP_CALL( SCIPincludeReaderSol(scip) );
   SCIP_CALL( SCIPincludeReaderZpl(scip) );
   SCIP_CALL( SCIPincludePresolDualfix(scip) );
   SCIP_CALL( SCIPincludePresolImplics(scip) );
   SCIP_CALL( SCIPincludePresolInttobinary(scip) );
   SCIP_CALL( SCIPincludePresolProbing(scip) );
   SCIP_CALL( SCIPincludePresolTrivial(scip) );
   SCIP_CALL( SCIPincludeNodeselBfs(scip) );
   SCIP_CALL( SCIPincludeNodeselDfs(scip) );
   SCIP_CALL( SCIPincludeNodeselEstimate(scip) );
   SCIP_CALL( SCIPincludeNodeselHybridestim(scip) );
   SCIP_CALL( SCIPincludeNodeselRestartdfs(scip) );
   SCIP_CALL( SCIPincludeBranchruleAllfullstrong(scip) );
   SCIP_CALL( SCIPincludeBranchruleFullstrong(scip) );
   SCIP_CALL( SCIPincludeBranchruleInference(scip) );
   SCIP_CALL( SCIPincludeBranchruleMostinf(scip) );
   SCIP_CALL( SCIPincludeBranchruleLeastinf(scip) );
   SCIP_CALL( SCIPincludeBranchrulePscost(scip) );
   SCIP_CALL( SCIPincludeBranchruleRandom(scip) );
   SCIP_CALL( SCIPincludeBranchruleRelpscost(scip) );
   SCIP_CALL( SCIPincludeHeurActconsdiving(scip) );
   SCIP_CALL( SCIPincludeHeurCoefdiving(scip) );
   SCIP_CALL( SCIPincludeHeurCrossover(scip) );
   SCIP_CALL( SCIPincludeHeurFeaspump(scip) );
   SCIP_CALL( SCIPincludeHeurFixandinfer(scip) );
   SCIP_CALL( SCIPincludeHeurFracdiving(scip) );
   SCIP_CALL( SCIPincludeHeurGuideddiving(scip) );
   SCIP_CALL( SCIPincludeHeurIntdiving(scip) );
   SCIP_CALL( SCIPincludeHeurIntshifting(scip) );
   SCIP_CALL( SCIPincludeHeurLinesearchdiving(scip) );
   SCIP_CALL( SCIPincludeHeurLocalbranching(scip) );
   SCIP_CALL( SCIPincludeHeurMutation(scip) );
   SCIP_CALL( SCIPincludeHeurObjpscostdiving(scip) );
   SCIP_CALL( SCIPincludeHeurOctane(scip) );
   SCIP_CALL( SCIPincludeHeurOneopt(scip) );
   SCIP_CALL( SCIPincludeHeurPscostdiving(scip) );
   SCIP_CALL( SCIPincludeHeurRens(scip) );
   SCIP_CALL( SCIPincludeHeurRins(scip) );
   SCIP_CALL( SCIPincludeHeurRootsoldiving(scip) );
   SCIP_CALL( SCIPincludeHeurRounding(scip) );
   SCIP_CALL( SCIPincludeHeurShifting(scip) );
   SCIP_CALL( SCIPincludeHeurSimplerounding(scip) );
   SCIP_CALL( SCIPincludeHeurVeclendiving(scip) );
   SCIP_CALL( SCIPincludePropPseudoobj(scip) );
   SCIP_CALL( SCIPincludePropRootredcost(scip) );
   SCIP_CALL( SCIPincludeSepaClique(scip) );
   SCIP_CALL( SCIPincludeSepaCmir(scip) );
   SCIP_CALL( SCIPincludeSepaFlowcover(scip) );
   SCIP_CALL( SCIPincludeSepaGomory(scip) );
   SCIP_CALL( SCIPincludeSepaImpliedbounds(scip) );
   SCIP_CALL( SCIPincludeSepaIntobj(scip) );
   SCIP_CALL( SCIPincludeSepaMcf(scip) );
   SCIP_CALL( SCIPincludeSepaRedcost(scip) );
   SCIP_CALL( SCIPincludeSepaStrongcg(scip) );
   SCIP_CALL( SCIPincludeDispDefault(scip) );
   SCIP_CALL( SCIPincludeDialogDefault(scip) );

   SCIP_CALL( SCIPdebugIncludeProp(scip) ); /*lint !e506 !e774*/

   return SCIP_OKAY;
}
