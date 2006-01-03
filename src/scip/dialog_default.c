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
#pragma ident "@(#) $Id: dialog_default.c,v 1.57 2006/01/03 12:22:46 bzfpfend Exp $"

/**@file   dialog_default.c
 * @brief  default user interface dialog
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "scip/dialog_default.h"



/** executes a menu dialog */
static
SCIP_RETCODE dialogExecMenu(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG*          dialog,             /**< dialog menu */
   SCIP_DIALOGHDLR*      dialoghdlr,         /**< dialog handler */
   SCIP_DIALOG**         nextdialog          /**< pointer to store next dialog to execute */
   )
{
   const char* command;
   SCIP_Bool again;
   int nfound;

   do
   {
      again = FALSE;

      /* get the next word of the command string */
      command = SCIPdialoghdlrGetWord(dialoghdlr, dialog, NULL);
      
      /* exit to the root dialog, if command is empty */
      if( command[0] == '\0' )
      {
         *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);
         return SCIP_OKAY;
      }
      else if( strcmp(command, "..") == 0 )
      {
         *nextdialog = SCIPdialogGetParent(dialog);
         if( *nextdialog == NULL )
            *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);
         return SCIP_OKAY;
      }
      
      /* find command in dialog */
      nfound = SCIPdialogFindEntry(dialog, command, nextdialog);
      
      /* check result */
      if( nfound == 0 )
      {
         SCIPdialogMessage(scip, NULL, "command <%s> not available\n", command);
         SCIPdialoghdlrClearBuffer(dialoghdlr);
         *nextdialog = dialog;
      }
      else if( nfound >= 2 )
      {
         SCIPdialogMessage(scip, NULL, "\npossible completions:\n");
         SCIP_CALL( SCIPdialogDisplayCompletions(dialog, scip, command) );
         SCIPdialogMessage(scip, NULL, "\n");
         SCIPdialoghdlrClearBuffer(dialoghdlr);
         again = TRUE;
      }
   }
   while( again );

   return SCIP_OKAY;
}

/** standard menu dialog execution method, that displays it's help screen if the remaining command line is empty */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecMenu)
{  /*lint --e{715}*/
   /* if remaining command string is empty, display menu of available options */
   if( SCIPdialoghdlrIsBufferEmpty(dialoghdlr) )
   {
      SCIPdialogMessage(scip, NULL, "\n");
      SCIP_CALL( SCIPdialogDisplayMenu(dialog, scip) );
      SCIPdialogMessage(scip, NULL, "\n");
   }

   SCIP_CALL( dialogExecMenu(scip, dialog, dialoghdlr, nextdialog) );

   return SCIP_OKAY;
}

/** standard menu dialog execution method, that doesn't display it's help screen */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecMenuLazy)
{  /*lint --e{715}*/
   SCIP_CALL( dialogExecMenu(scip, dialog, dialoghdlr, nextdialog) );

   return SCIP_OKAY;
}

/** dialog execution method for the checksol command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecChecksol)
{  /*lint --e{715}*/
   SCIP_SOL* sol;
   SCIP_CONSHDLR* infeasconshdlr;
   SCIP_CONS* infeascons;
   SCIP_Bool feasible;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED )
      sol = SCIPgetBestSol(scip);
   else
      sol = NULL;

   if( sol == NULL )
      SCIPdialogMessage(scip, NULL, "no feasible solution available\n");
   else
   {
      SCIP_CALL( SCIPcheckSolOrig(scip, sol, &feasible, &infeasconshdlr, &infeascons) );
      
      if( feasible )
         SCIPdialogMessage(scip, NULL, "best solution is feasible in original problem\n");
      else if( infeascons == NULL )
      { 
         SCIPdialogMessage(scip, NULL, "best solution violates constraint handler [%s]\n",
            SCIPconshdlrGetName(infeasconshdlr));
      }
      else
      { 
         SCIPdialogMessage(scip, NULL, "best solution violates constraint <%s> [%s] of original problem:\n", 
            SCIPconsGetName(infeascons), SCIPconshdlrGetName(infeasconshdlr));
         SCIP_CALL( SCIPprintCons(scip, infeascons, NULL) );
      }
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialogGetParent(dialog);

   return SCIP_OKAY;
}

/** dialog execution method for the conflictgraph command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecConflictgraph)
{  /*lint --e{715}*/
   SCIP_RETCODE retcode;
   const char* filename;

   if( !SCIPisTransformed(scip) )
   {
      SCIPdialogMessage(scip, NULL, "cannot call method before problem was transformed\n");
      SCIPdialoghdlrClearBuffer(dialoghdlr);
   }
   else
   {
      filename = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter filename: ");

      if( filename[0] != '\0' )
      {
         SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, filename) );

         retcode = SCIPwriteImplicationConflictGraph(scip, filename);
         if( retcode == SCIP_FILECREATEERROR )
            SCIPdialogMessage(scip, NULL, "error writing file <%s>\n", filename);
         else
         {
            SCIP_CALL( retcode );
         }
      }
   }

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display branching command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayBranching)
{  /*lint --e{715}*/
   SCIP_BRANCHRULE** branchrules;
   SCIP_BRANCHRULE** sorted;
   int nbranchrules;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   branchrules = SCIPgetBranchrules(scip);
   nbranchrules = SCIPgetNBranchrules(scip);

   /* copy branchrules array into temporary memory for sorting */
   SCIP_CALL( SCIPduplicateBufferArray(scip, &sorted, branchrules, nbranchrules) );

   /* sort the branching rules */
   SCIPbsortPtr((void**)sorted, nbranchrules, SCIPbranchruleComp);

   /* display sorted list of branching rules */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " branching rule       priority maxdepth maxbddist  description\n");
   SCIPdialogMessage(scip, NULL, " --------------       -------- -------- ---------  -----------\n");
   for( i = 0; i < nbranchrules; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPbranchruleGetName(sorted[i]));
      if( strlen(SCIPbranchruleGetName(sorted[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%8d %8d %8.1f%%  ", SCIPbranchruleGetPriority(sorted[i]), 
         SCIPbranchruleGetMaxdepth(sorted[i]), 100.0 * SCIPbranchruleGetMaxbounddist(sorted[i]));
      SCIPdialogMessage(scip, NULL, SCIPbranchruleGetDesc(sorted[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &sorted);

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display conflict command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayConflict)
{  /*lint --e{715}*/
   SCIP_CONFLICTHDLR** conflicthdlrs;
   SCIP_CONFLICTHDLR** sorted;
   int nconflicthdlrs;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   conflicthdlrs = SCIPgetConflicthdlrs(scip);
   nconflicthdlrs = SCIPgetNConflicthdlrs(scip);

   /* copy conflicthdlrs array into temporary memory for sorting */
   SCIP_CALL( SCIPduplicateBufferArray(scip, &sorted, conflicthdlrs, nconflicthdlrs) );

   /* sort the conflict handlers */
   SCIPbsortPtr((void**)sorted, nconflicthdlrs, SCIPconflicthdlrComp);

   /* display sorted list of conflict handlers */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " conflict handler     priority  description\n");
   SCIPdialogMessage(scip, NULL, " ----------------     --------  -----------\n");
   for( i = 0; i < nconflicthdlrs; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPconflicthdlrGetName(sorted[i]));
      if( strlen(SCIPconflicthdlrGetName(sorted[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%8d  ", SCIPconflicthdlrGetPriority(sorted[i]));
      SCIPdialogMessage(scip, NULL, SCIPconflicthdlrGetDesc(sorted[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &sorted);

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display conshdlrs command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayConshdlrs)
{  /*lint --e{715}*/
   SCIP_CONSHDLR** conshdlrs;
   int nconshdlrs;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   conshdlrs = SCIPgetConshdlrs(scip);
   nconshdlrs = SCIPgetNConshdlrs(scip);

   /* display list of constraint handlers */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " constraint handler   chckprio enfoprio sepaprio sepaf propf eager  description\n");
   SCIPdialogMessage(scip, NULL, " ------------------   -------- -------- -------- ----- ----- -----  -----------\n");
   for( i = 0; i < nconshdlrs; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPconshdlrGetName(conshdlrs[i]));
      if( strlen(SCIPconshdlrGetName(conshdlrs[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%8d %8d %8d %5d %5d %5d  ",
         SCIPconshdlrGetCheckPriority(conshdlrs[i]),
         SCIPconshdlrGetEnfoPriority(conshdlrs[i]),
         SCIPconshdlrGetSepaPriority(conshdlrs[i]),
         SCIPconshdlrGetSepaFreq(conshdlrs[i]),
         SCIPconshdlrGetPropFreq(conshdlrs[i]),
         SCIPconshdlrGetEagerFreq(conshdlrs[i]));
      SCIPdialogMessage(scip, NULL, SCIPconshdlrGetDesc(conshdlrs[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display displaycols command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayDisplaycols)
{  /*lint --e{715}*/
   SCIP_DISP** disps;
   int ndisps;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   disps = SCIPgetDisps(scip);
   ndisps = SCIPgetNDisps(scip);

   /* display list of display columns */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " display column       header           position width priority status  description\n");
   SCIPdialogMessage(scip, NULL, " --------------       ------           -------- ----- -------- ------  -----------\n");
   for( i = 0; i < ndisps; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPdispGetName(disps[i]));
      if( strlen(SCIPdispGetName(disps[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%-16s ", SCIPdispGetHeader(disps[i]));
      if( strlen(SCIPdispGetHeader(disps[i])) > 16 )
         SCIPdialogMessage(scip, NULL, "\n %20s %16s ", "", "-->");
      SCIPdialogMessage(scip, NULL, "%8d ", SCIPdispGetPosition(disps[i]));
      SCIPdialogMessage(scip, NULL, "%5d ", SCIPdispGetWidth(disps[i]));
      SCIPdialogMessage(scip, NULL, "%8d ", SCIPdispGetPriority(disps[i]));
      switch( SCIPdispGetStatus(disps[i]) )
      {
      case SCIP_DISPSTATUS_OFF:
         SCIPdialogMessage(scip, NULL, "%6s  ", "off");
         break;
      case SCIP_DISPSTATUS_AUTO:
         SCIPdialogMessage(scip, NULL, "%6s  ", "auto");
         break;
      case SCIP_DISPSTATUS_ON:
         SCIPdialogMessage(scip, NULL, "%6s  ", "on");
         break;
      default:
         SCIPdialogMessage(scip, NULL, "%6s  ", "???");
         break;
      }
      SCIPdialogMessage(scip, NULL, SCIPdispGetDesc(disps[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display heuristics command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayHeuristics)
{  /*lint --e{715}*/
   SCIP_HEUR** heurs;
   int nheurs;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   heurs = SCIPgetHeurs(scip);
   nheurs = SCIPgetNHeurs(scip);

   /* display list of primal heuristics */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " primal heuristic     c priority freq ofs  description\n");
   SCIPdialogMessage(scip, NULL, " ----------------     - -------- ---- ---  -----------\n");
   for( i = 0; i < nheurs; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPheurGetName(heurs[i]));
      if( strlen(SCIPheurGetName(heurs[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%c ", SCIPheurGetDispchar(heurs[i]));
      SCIPdialogMessage(scip, NULL, "%8d ", SCIPheurGetPriority(heurs[i]));
      SCIPdialogMessage(scip, NULL, "%4d ", SCIPheurGetFreq(heurs[i]));
      SCIPdialogMessage(scip, NULL, "%3d  ", SCIPheurGetFreqofs(heurs[i]));
      SCIPdialogMessage(scip, NULL, SCIPheurGetDesc(heurs[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display memory command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayMemory)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   SCIPprintMemoryDiagnostic(scip);
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display nodeselectors command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayNodeselectors)
{  /*lint --e{715}*/
   SCIP_NODESEL** nodesels;
   int nnodesels;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   nodesels = SCIPgetNodesels(scip);
   nnodesels = SCIPgetNNodesels(scip);

   /* display list of node selectors */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " node selector        std priority memsave prio  description\n");
   SCIPdialogMessage(scip, NULL, " -------------        ------------ ------------  -----------\n");
   for( i = 0; i < nnodesels; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPnodeselGetName(nodesels[i]));
      if( strlen(SCIPnodeselGetName(nodesels[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%12d ", SCIPnodeselGetStdPriority(nodesels[i]));
      SCIPdialogMessage(scip, NULL, "%12d  ", SCIPnodeselGetMemsavePriority(nodesels[i]));
      SCIPdialogMessage(scip, NULL, SCIPnodeselGetDesc(nodesels[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display presolvers command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayPresolvers)
{  /*lint --e{715}*/
   SCIP_PRESOL** presols;
   int npresols;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   presols = SCIPgetPresols(scip);
   npresols = SCIPgetNPresols(scip);

   /* display list of presolvers */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " presolver            priority  description\n");
   SCIPdialogMessage(scip, NULL, " ---------            --------  -----------\n");
   for( i = 0; i < npresols; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPpresolGetName(presols[i]));
      if( strlen(SCIPpresolGetName(presols[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%8d%c ", SCIPpresolGetPriority(presols[i]),
         SCIPpresolIsDelayed(presols[i]) ? 'd' : ' ');
      SCIPdialogMessage(scip, NULL, SCIPpresolGetDesc(presols[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display problem command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayProblem)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   SCIP_CALL( SCIPprintOrigProblem(scip, NULL) );
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display propagators command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayPropagators)
{  /*lint --e{715}*/
   SCIP_PROP** props;
   int nprops;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   props = SCIPgetProps(scip);
   nprops = SCIPgetNProps(scip);

   /* display list of propagators */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " propagator           priority  freq  description\n");
   SCIPdialogMessage(scip, NULL, " ----------           --------  ----  -----------\n");
   for( i = 0; i < nprops; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPpropGetName(props[i]));
      if( strlen(SCIPpropGetName(props[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%8d%c ", SCIPpropGetPriority(props[i]), SCIPpropIsDelayed(props[i]) ? 'd' : ' ');
      SCIPdialogMessage(scip, NULL, "%4d  ", SCIPpropGetFreq(props[i]));
      SCIPdialogMessage(scip, NULL, SCIPpropGetDesc(props[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display readers command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayReaders)
{  /*lint --e{715}*/
   SCIP_READER** readers;
   int nreaders;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   readers = SCIPgetReaders(scip);
   nreaders = SCIPgetNReaders(scip);

   /* display list of readers */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " file reader          extension  description\n");
   SCIPdialogMessage(scip, NULL, " -----------          ---------  -----------\n");
   for( i = 0; i < nreaders; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPreaderGetName(readers[i]));
      if( strlen(SCIPreaderGetName(readers[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%9s  ", SCIPreaderGetExtension(readers[i]));
      SCIPdialogMessage(scip, NULL, SCIPreaderGetDesc(readers[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display separators command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplaySeparators)
{  /*lint --e{715}*/
   SCIP_SEPA** sepas;
   int nsepas;
   int i;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   sepas = SCIPgetSepas(scip);
   nsepas = SCIPgetNSepas(scip);

   /* display list of separators */
   SCIPdialogMessage(scip, NULL, "\n");
   SCIPdialogMessage(scip, NULL, " separator            priority  freq  description\n");
   SCIPdialogMessage(scip, NULL, " ---------            --------  ----  -----------\n");
   for( i = 0; i < nsepas; ++i )
   {
      SCIPdialogMessage(scip, NULL, " %-20s ", SCIPsepaGetName(sepas[i]));
      if( strlen(SCIPsepaGetName(sepas[i])) > 20 )
         SCIPdialogMessage(scip, NULL, "\n %20s ", "-->");
      SCIPdialogMessage(scip, NULL, "%8d%c ", SCIPsepaGetPriority(sepas[i]), SCIPsepaIsDelayed(sepas[i]) ? 'd' : ' ');
      SCIPdialogMessage(scip, NULL, "%4d  ", SCIPsepaGetFreq(sepas[i]));
      SCIPdialogMessage(scip, NULL, SCIPsepaGetDesc(sepas[i]));
      SCIPdialogMessage(scip, NULL, "\n");
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display solution command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplaySolution)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   SCIP_CALL( SCIPprintBestSol(scip, NULL, FALSE) );
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display statistics command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayStatistics)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   SCIP_CALL( SCIPprintStatistics(scip, NULL) );
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display transproblem command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayTransproblem)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   SCIP_CALL( SCIPprintTransProblem(scip, NULL) );
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display value command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayValue)
{  /*lint --e{715}*/
   SCIP_SOL* sol;
   SCIP_VAR* var;
   const char* varname;
   SCIP_Real solval;

   SCIPdialogMessage(scip, NULL, "\n");

   if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED )
      sol = SCIPgetBestSol(scip);
   else
      sol = NULL;

   if( sol == NULL )
   {
      SCIPdialogMessage(scip, NULL, "no feasible solution available\n");
      SCIPdialoghdlrClearBuffer(dialoghdlr);
   }
   else
   {
      varname = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter variable name: ");

      if( varname[0] != '\0' )
      {
         SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, varname) );

         var = SCIPfindVar(scip, varname);
         if( var == NULL )
            SCIPdialogMessage(scip, NULL, "variable <%s> not found\n", varname);
         else
         {
            solval = SCIPgetSolVal(scip, sol, var);
            SCIPdialogMessage(scip, NULL, "%-32s", SCIPvarGetName(var));
            if( SCIPisInfinity(scip, solval) )
               SCIPdialogMessage(scip, NULL, " +infinity");
            else if( SCIPisInfinity(scip, -solval) )
               SCIPdialogMessage(scip, NULL, " -infinity");
            else
               SCIPdialogMessage(scip, NULL, " %f", solval);
            SCIPdialogMessage(scip, NULL, " \t(obj:%g)\n", SCIPvarGetObj(var));
         }
      }
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the display varbranchstatistics command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayVarbranchstatistics)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   SCIP_CALL( SCIPprintBranchingStatistics(scip, NULL) );
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the help command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecHelp)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   SCIP_CALL( SCIPdialogDisplayMenu(SCIPdialogGetParent(dialog), scip) );
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialogGetParent(dialog);

   return SCIP_OKAY;
}

/** dialog execution method for the display transsolution command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecDisplayTranssolution)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED )
   {
      if( SCIPsolGetOrigin(SCIPgetBestSol(scip)) == SCIP_SOLORIGIN_ORIGINAL )
      {
         SCIPdialogMessage(scip, NULL, "best solution exists only in original problem space\n");
      }
      else
      {
         SCIP_CALL( SCIPprintBestTransSol(scip, NULL, FALSE) );
      }
   }
   else
      SCIPdialogMessage(scip, NULL, "no solution available\n");
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the free command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecFree)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIP_CALL( SCIPfreeProb(scip) );

   *nextdialog = SCIPdialogGetParent(dialog);

   return SCIP_OKAY;
}

/** dialog execution method for the newstart command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecNewstart)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIP_CALL( SCIPfreeSolve(scip) );

   *nextdialog = SCIPdialogGetParent(dialog);

   return SCIP_OKAY;
}

/** dialog execution method for the optimize command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecOptimize)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   switch( SCIPgetStage(scip) )
   {
   case SCIP_STAGE_INIT:
      SCIPdialogMessage(scip, NULL, "no problem exists\n");
      break;

   case SCIP_STAGE_PROBLEM:
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPsolve(scip) );
      break;

   case SCIP_STAGE_SOLVED:
      SCIPdialogMessage(scip, NULL, "problem is already solved\n");
      break;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
   default:
      SCIPerrorMessage("invalid SCIP stage\n");
      return SCIP_INVALIDCALL;
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the presolve command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecPresolve)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL) );

   SCIPdialogMessage(scip, NULL, "\n");
   switch( SCIPgetStage(scip) )
   {
   case SCIP_STAGE_INIT:
      SCIPdialogMessage(scip, NULL, "no problem exists\n");
      break;

   case SCIP_STAGE_PROBLEM:
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
      SCIP_CALL( SCIPpresolve(scip) );
      break;

   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      SCIPdialogMessage(scip, NULL, "problem is already presolved\n");
      break;

   case SCIP_STAGE_SOLVED:
      SCIPdialogMessage(scip, NULL, "problem is already solved\n");
      break;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
   default:
      SCIPerrorMessage("invalid SCIP stage\n");
      return SCIP_INVALIDCALL;
   }
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the quit command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecQuit)
{  /*lint --e{715}*/
   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = NULL;

   return SCIP_OKAY;
}

/** dialog execution method for the read command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecRead)
{  /*lint --e{715}*/
   SCIP_RETCODE retcode;
   const char* filename;

   filename = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter filename: ");

   if( filename[0] != '\0' )
   {
      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, filename) );

      if( SCIPfileExists(filename) )
      {
         retcode = SCIPreadProb(scip, filename);
         if( retcode == SCIP_READERROR || retcode == SCIP_NOFILE || retcode == SCIP_PARSEERROR )
         {
            SCIPdialogMessage(scip, NULL, "error reading file <%s>\n", filename);
            SCIP_CALL( SCIPfreeProb(scip) );
         }
         else
         {
            SCIP_CALL( retcode );
         }
      }
      else
      {
         SCIPdialogMessage(scip, NULL, "file <%s> not found\n", filename);
         SCIPdialoghdlrClearBuffer(dialoghdlr);
      }
   }

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the set load command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetLoad)
{  /*lint --e{715}*/
   const char* filename;

   filename = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter filename: ");

   if( filename[0] != '\0' )
   {
      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, filename) );

      if( SCIPfileExists(filename) )
      {
         SCIP_CALL( SCIPreadParams(scip, filename) );
         SCIPdialogMessage(scip, NULL, "loaded parameter file <%s>\n", filename);
      }
      else
      {
         SCIPdialogMessage(scip, NULL, "file <%s> not found\n", filename);
         SCIPdialoghdlrClearBuffer(dialoghdlr);
      }
   }

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the set save command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetSave)
{  /*lint --e{715}*/
   const char* filename;

   filename = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter filename: ");

   if( filename[0] != '\0' )
   {
      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, filename) );

      SCIP_CALL( SCIPwriteParams(scip, filename, TRUE, FALSE) );
      SCIPdialogMessage(scip, NULL, "saved parameter file <%s>\n", filename);
   }

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the set diffsave command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetDiffsave)
{  /*lint --e{715}*/
   const char* filename;

   filename = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter filename: ");

   if( filename[0] != '\0' )
   {
      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, filename) );
   
      SCIP_CALL( SCIPwriteParams(scip, filename, TRUE, TRUE) );
      SCIPdialogMessage(scip, NULL, "saved non-default parameter settings to file <%s>\n", filename);
   }

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the set parameter command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetParam)
{  /*lint --e{715}*/
   SCIP_RETCODE retcode;
   SCIP_PARAM* param;
   char prompt[SCIP_MAXSTRLEN];
   const char* valuestr;
   int intval;
   SCIP_Longint longintval;
   SCIP_Real realval;
   char charval;

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   /* get the parameter to set */
   param = (SCIP_PARAM*)SCIPdialogGetData(dialog);
   
   /* depending on the parameter type, request a user input */
   switch( SCIPparamGetType(param) )
   {
   case SCIP_PARAMTYPE_BOOL:
      snprintf(prompt, SCIP_MAXSTRLEN, "current value: %s, new value (TRUE/FALSE): ",
         SCIPparamGetBool(param) ? "TRUE" : "FALSE");
      valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, prompt);
      if( valuestr[0] == '\0' )
         return SCIP_OKAY;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, valuestr) );

      switch( valuestr[0] )
      {
      case 'f':
      case 'F':
      case '0':
      case 'n':
      case 'N':
         SCIP_CALL( SCIPparamSetBool(param, scip, FALSE) );
         SCIPdialogMessage(scip, NULL, "parameter <%s> set to FALSE\n", SCIPparamGetName(param));
         break;
      case 't':
      case 'T':
      case '1':
      case 'y':
      case 'Y':
         SCIP_CALL( SCIPparamSetBool(param, scip, TRUE) );
         SCIPdialogMessage(scip, NULL, "parameter <%s> set to TRUE\n", SCIPparamGetName(param));
         break;
      default:
         SCIPdialogMessage(scip, NULL, "\ninvalid parameter value <%s>\n\n", valuestr);
         break;
      }
      break;

   case SCIP_PARAMTYPE_INT:
      snprintf(prompt, SCIP_MAXSTRLEN, "current value: %d, new value [%d,%d]: ",
         SCIPparamGetInt(param), SCIPparamGetIntMin(param), SCIPparamGetIntMax(param));
      valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, prompt);
      if( valuestr[0] == '\0' )
         return SCIP_OKAY;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, valuestr) );

      if( sscanf(valuestr, "%d", &intval) != 1 )
      {
         SCIPdialogMessage(scip, NULL, "\ninvalid input <%s>\n\n", valuestr);
         return SCIP_OKAY;
      }
      retcode = SCIPparamSetInt(param, scip, intval);
      if( retcode != SCIP_PARAMETERWRONGVAL )
      {
         SCIP_CALL( retcode );
      }
      SCIPdialogMessage(scip, NULL, "parameter <%s> set to %d\n", SCIPparamGetName(param), SCIPparamGetInt(param));
      break;

   case SCIP_PARAMTYPE_LONGINT:
      snprintf(prompt, SCIP_MAXSTRLEN, "current value: %"SCIP_LONGINT_FORMAT", new value [%"SCIP_LONGINT_FORMAT",%"SCIP_LONGINT_FORMAT"]: ",
         SCIPparamGetLongint(param), SCIPparamGetLongintMin(param), SCIPparamGetLongintMax(param));
      valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, prompt);
      if( valuestr[0] == '\0' )
         return SCIP_OKAY;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, valuestr) );

      if( sscanf(valuestr, "%"SCIP_LONGINT_FORMAT, &longintval) != 1 )
      {
         SCIPdialogMessage(scip, NULL, "\ninvalid input <%s>\n\n", valuestr);
         return SCIP_OKAY;
      }
      retcode = SCIPparamSetLongint(param, scip, longintval);
      if( retcode != SCIP_PARAMETERWRONGVAL )
      {
         SCIP_CALL( retcode );
      }
      SCIPdialogMessage(scip, NULL, "parameter <%s> set to %"SCIP_LONGINT_FORMAT"\n", SCIPparamGetName(param), SCIPparamGetLongint(param));
      break;

   case SCIP_PARAMTYPE_REAL:
      snprintf(prompt, SCIP_MAXSTRLEN, "current value: %g, new value [%g,%g]: ",
         SCIPparamGetReal(param), SCIPparamGetRealMin(param), SCIPparamGetRealMax(param));
      valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, prompt);
      if( valuestr[0] == '\0' )
         return SCIP_OKAY;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, valuestr) );

      if( sscanf(valuestr, "%"SCIP_REAL_FORMAT, &realval) != 1 )
      {
         SCIPdialogMessage(scip, NULL, "\ninvalid input <%s>\n\n", valuestr);
         return SCIP_OKAY;
      }
      retcode = SCIPparamSetReal(param, scip, realval);
      if( retcode != SCIP_PARAMETERWRONGVAL )
      {
         SCIP_CALL( retcode );
      }
      SCIPdialogMessage(scip, NULL, "parameter <%s> set to %g\n", SCIPparamGetName(param), SCIPparamGetReal(param));
      break;

   case SCIP_PARAMTYPE_CHAR:
      snprintf(prompt, SCIP_MAXSTRLEN, "current value: <%c>, new value: ", SCIPparamGetChar(param));
      valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, prompt);
      if( valuestr[0] == '\0' )
         return SCIP_OKAY;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, valuestr) );

      if( sscanf(valuestr, "%c", &charval) != 1 )
      {
         SCIPdialogMessage(scip, NULL, "\ninvalid input <%s>\n\n", valuestr);
         return SCIP_OKAY;
      }
      retcode = SCIPparamSetChar(param, scip, charval);
      if( retcode != SCIP_PARAMETERWRONGVAL )
      {
         SCIP_CALL( retcode );
      }
      SCIPdialogMessage(scip, NULL, "parameter <%s> set to <%c>\n", SCIPparamGetName(param), SCIPparamGetChar(param));
      break;

   case SCIP_PARAMTYPE_STRING:
      snprintf(prompt, SCIP_MAXSTRLEN, "current value: <%s>, new value: ", SCIPparamGetString(param));
      valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, prompt);
      if( valuestr[0] == '\0' )
         return SCIP_OKAY;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, valuestr) );

      retcode = SCIPparamSetString(param, scip, valuestr);
      if( retcode != SCIP_PARAMETERWRONGVAL )
      {
         SCIP_CALL( retcode );
      }
      SCIPdialogMessage(scip, NULL, "parameter <%s> set to <%s>\n", SCIPparamGetName(param), SCIPparamGetString(param));
      break;

   default:
      SCIPerrorMessage("invalid parameter type\n");
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}

/** dialog description method for the set parameter command */
SCIP_DECL_DIALOGDESC(SCIPdialogDescSetParam)
{  /*lint --e{715}*/
   SCIP_PARAM* param;
   char valuestr[SCIP_MAXSTRLEN];

   /* get the parameter to set */
   param = (SCIP_PARAM*)SCIPdialogGetData(dialog);
   
   /* retrieve parameter's current value */
   switch( SCIPparamGetType(param) )
   {
   case SCIP_PARAMTYPE_BOOL:
      if( SCIPparamGetBool(param) )
         sprintf(valuestr, "TRUE");
      else
         sprintf(valuestr, "FALSE");
      break;

   case SCIP_PARAMTYPE_INT:
      sprintf(valuestr, "%d", SCIPparamGetInt(param));
      break;

   case SCIP_PARAMTYPE_LONGINT:
      sprintf(valuestr, "%"SCIP_LONGINT_FORMAT, SCIPparamGetLongint(param));
      break;

   case SCIP_PARAMTYPE_REAL:
      sprintf(valuestr, "%g", SCIPparamGetReal(param));
      if( strchr(valuestr, '.') == NULL && strchr(valuestr, 'e') == NULL )
         sprintf(valuestr, "%.1f", SCIPparamGetReal(param));
      break;

   case SCIP_PARAMTYPE_CHAR:
      sprintf(valuestr, "%c", SCIPparamGetChar(param));
      break;

   case SCIP_PARAMTYPE_STRING:
      snprintf(valuestr, SCIP_MAXSTRLEN, "%s", SCIPparamGetString(param));
      break;

   default:
      SCIPerrorMessage("invalid parameter type\n");
      return SCIP_INVALIDDATA;
   }
   valuestr[SCIP_MAXSTRLEN-1] = '\0';

   /* display parameter's description */
   SCIPdialogMessage(scip, NULL, SCIPparamGetDesc(param));

   /* display parameter's current value */
   SCIPdialogMessage(scip, NULL, " [%s]", valuestr);
   
   return SCIP_OKAY;
}

/** dialog execution method for the set branching direction command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetBranchingDirection)
{  /*lint --e{715}*/
   SCIP_VAR* var;
   char prompt[SCIP_MAXSTRLEN];
   const char* valuestr;
   int direction;

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   /* branching priorities cannot be set, if no problem was created */
   if( SCIPgetStage(scip) == SCIP_STAGE_INIT )
   {
      SCIPdialogMessage(scip, NULL, "cannot set branching directions before problem was created\n");
      return SCIP_OKAY;
   }

   /* get variable name from user */
   valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "variable name: ");
   if( valuestr[0] == '\0' )
      return SCIP_OKAY;

   /* find variable */
   var = SCIPfindVar(scip, valuestr);
   if( var == NULL )
   {
      SCIPdialogMessage(scip, NULL, "variable <%s> does not exist in problem\n", valuestr);
      return SCIP_OKAY;
   }

   /* get new branching direction from user */
   switch( SCIPvarGetBranchDirection(var) )
   {
   case SCIP_BRANCHDIR_DOWNWARDS:
      direction = -1;
      break;
   case SCIP_BRANCHDIR_AUTO:
      direction = 0;
      break;
   case SCIP_BRANCHDIR_UPWARDS:
      direction = +1;
      break;
   default:
      SCIPerrorMessage("invalid preferred branching direction <%d> of variable <%s>\n", 
         SCIPvarGetBranchDirection(var), SCIPvarGetName(var));
      return SCIP_INVALIDDATA;
   }
   snprintf(prompt, SCIP_MAXSTRLEN, "current value: %d, new value: ", direction);
   valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, prompt);
   snprintf(prompt, SCIP_MAXSTRLEN, "%s %s", SCIPvarGetName(var), valuestr);
   if( valuestr[0] == '\0' )
      return SCIP_OKAY;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, prompt) );

   if( sscanf(valuestr, "%d", &direction) != 1 )
   {
      SCIPdialogMessage(scip, NULL, "\ninvalid input <%s>\n\n", valuestr);
      return SCIP_OKAY;
   }
   if( direction < -1 || direction > +1 )
   {
      SCIPdialogMessage(scip, NULL, "\ninvalid input <%d>: direction must be -1, 0, or +1\n\n", direction);
      return SCIP_OKAY;
   }

   /* set new branching direction */
   if( direction == -1 )
      SCIP_CALL( SCIPchgVarBranchDirection(scip, var, SCIP_BRANCHDIR_DOWNWARDS) );
   else if( direction == 0 )
      SCIP_CALL( SCIPchgVarBranchDirection(scip, var, SCIP_BRANCHDIR_AUTO) );
   else
      SCIP_CALL( SCIPchgVarBranchDirection(scip, var, SCIP_BRANCHDIR_UPWARDS) );

   SCIPdialogMessage(scip, NULL, "branching direction of variable <%s> set to %d\n", SCIPvarGetName(var), direction);

   return SCIP_OKAY;
}

/** dialog execution method for the set branching priority command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetBranchingPriority)
{  /*lint --e{715}*/
   SCIP_VAR* var;
   char prompt[SCIP_MAXSTRLEN];
   const char* valuestr;
   int priority;

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   /* branching priorities cannot be set, if no problem was created */
   if( SCIPgetStage(scip) == SCIP_STAGE_INIT )
   {
      SCIPdialogMessage(scip, NULL, "cannot set branching priorities before problem was created\n");
      return SCIP_OKAY;
   }

   /* get variable name from user */
   valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "variable name: ");
   if( valuestr[0] == '\0' )
      return SCIP_OKAY;

   /* find variable */
   var = SCIPfindVar(scip, valuestr);
   if( var == NULL )
   {
      SCIPdialogMessage(scip, NULL, "variable <%s> does not exist in problem\n", valuestr);
      return SCIP_OKAY;
   }

   /* get new branching priority from user */
   snprintf(prompt, SCIP_MAXSTRLEN, "current value: %d, new value: ", SCIPvarGetBranchPriority(var));
   valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, prompt);
   snprintf(prompt, SCIP_MAXSTRLEN, "%s %s", SCIPvarGetName(var), valuestr);
   if( valuestr[0] == '\0' )
      return SCIP_OKAY;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, prompt) );

   if( sscanf(valuestr, "%d", &priority) != 1 )
   {
      SCIPdialogMessage(scip, NULL, "\ninvalid input <%s>\n\n", valuestr);
      return SCIP_OKAY;
   }
   
   /* set new branching priority */
   SCIP_CALL( SCIPchgVarBranchPriority(scip, var, priority) );
   SCIPdialogMessage(scip, NULL, "branching priority of variable <%s> set to %d\n", SCIPvarGetName(var), SCIPvarGetBranchPriority(var));

   return SCIP_OKAY;
}

/** dialog execution method for the set limits objective command */
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetLimitsObjective)
{  /*lint --e{715}*/
   char prompt[SCIP_MAXSTRLEN];
   const char* valuestr;
   SCIP_Real objlim;

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   /* objective limit cannot be set, if no problem was created */
   if( SCIPgetStage(scip) == SCIP_STAGE_INIT )
   {
      SCIPdialogMessage(scip, NULL, "cannot set objective limit before problem was created\n");
      return SCIP_OKAY;
   }

   /* get new objective limit from user */
   snprintf(prompt, SCIP_MAXSTRLEN, "current value: %g, new value: ", SCIPgetObjlimit(scip));
   valuestr = SCIPdialoghdlrGetWord(dialoghdlr, dialog, prompt);
   if( valuestr[0] == '\0' )
      return SCIP_OKAY;

   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, valuestr) );

   if( sscanf(valuestr, "%"SCIP_REAL_FORMAT, &objlim) != 1 )
   {
      SCIPdialogMessage(scip, NULL, "\ninvalid input <%s>\n\n", valuestr);
      return SCIP_OKAY;
   }
   
   /* check, if new objective limit is valid */
   if( SCIPgetStage(scip) > SCIP_STAGE_PROBLEM
      && SCIPtransformObj(scip, objlim) > SCIPtransformObj(scip, SCIPgetObjlimit(scip)) )
   {
      SCIPdialogMessage(scip, NULL, "\ncannot relax objective limit from %g to %g after problem was transformed\n\n",
         SCIPgetObjlimit(scip), objlim);
      return SCIP_OKAY;
   }

   /* set new objective limit */
   SCIP_CALL( SCIPsetObjlimit(scip, objlim) );
   SCIPdialogMessage(scip, NULL, "objective value limit set to %g\n", SCIPgetObjlimit(scip));

   return SCIP_OKAY;
}

/** dialog execution method for the write problem command */
static
SCIP_DECL_DIALOGEXEC(SCIPdialogExecWriteProblem)
{  /*lint --e{715}*/
   const char* filename;

   SCIPdialogMessage(scip, NULL, "\n");

   filename = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter filename: ");
   if( filename[0] != '\0' )
   {
      FILE* file;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, filename) );

      file = fopen(filename, "w");
      if( file == NULL )
      {
         SCIPdialogMessage(scip, NULL, "error creating file <%s>\n", filename);
         SCIPdialoghdlrClearBuffer(dialoghdlr);
      }
      else
      {
         SCIP_CALL( SCIPprintOrigProblem(scip, file) );
         SCIPdialogMessage(scip, NULL, "written original problem to file <%s>\n", filename);
         fclose(file);
      }
   }

   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the write solution command */
static
SCIP_DECL_DIALOGEXEC(SCIPdialogExecWriteSolution)
{  /*lint --e{715}*/
   const char* filename;

   SCIPdialogMessage(scip, NULL, "\n");

   filename = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter filename: ");
   if( filename[0] != '\0' )
   {
      FILE* file;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, filename) );

      file = fopen(filename, "w");
      if( file == NULL )
      {
         SCIPdialogMessage(scip, NULL, "error creating file <%s>\n", filename);
         SCIPdialoghdlrClearBuffer(dialoghdlr);
      }
      else
      {
         SCIPinfoMessage(scip, file, "solution status: ");
         SCIP_CALL( SCIPprintStatus(scip, file) );
         SCIPinfoMessage(scip, file, "\n");
         SCIP_CALL( SCIPprintBestSol(scip, file, FALSE) );
         SCIPdialogMessage(scip, NULL, "written solution information to file <%s>\n", filename);
         fclose(file);
      }
   }

   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the write statistics command */
static
SCIP_DECL_DIALOGEXEC(SCIPdialogExecWriteStatistics)
{  /*lint --e{715}*/
   const char* filename;

   SCIPdialogMessage(scip, NULL, "\n");

   filename = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter filename: ");
   if( filename[0] != '\0' )
   {
      FILE* file;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, filename) );

      file = fopen(filename, "w");
      if( file == NULL )
      {
         SCIPdialogMessage(scip, NULL, "error creating file <%s>\n", filename);
         SCIPdialoghdlrClearBuffer(dialoghdlr);
      }
      else
      {
         SCIP_CALL( SCIPprintStatistics(scip, file) );
         SCIPdialogMessage(scip, NULL, "written statistics to file <%s>\n", filename);
         fclose(file);
      }
   }

   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** dialog execution method for the write transproblem command */
static
SCIP_DECL_DIALOGEXEC(SCIPdialogExecWriteTransproblem)
{  /*lint --e{715}*/
   const char* filename;

   SCIPdialogMessage(scip, NULL, "\n");

   filename = SCIPdialoghdlrGetWord(dialoghdlr, dialog, "enter filename: ");
   if( filename[0] != '\0' )
   {
      FILE* file;

      SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, filename) );

      file = fopen(filename, "w");
      if( file == NULL )
      {
         SCIPdialogMessage(scip, NULL, "error creating file <%s>\n", filename);
         SCIPdialoghdlrClearBuffer(dialoghdlr);
      }
      else
      {
         SCIP_CALL( SCIPprintTransProblem(scip, file) );
         SCIPdialogMessage(scip, NULL, "written transformed problem to file <%s>\n", filename);
         fclose(file);
      }
   }

   SCIPdialogMessage(scip, NULL, "\n");

   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}

/** includes or updates the default dialog menus in SCIP */
SCIP_RETCODE SCIPincludeDialogDefault(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_DIALOG* root;
   SCIP_DIALOG* submenu;
   SCIP_DIALOG* dialog;

   /* root menu */
   root = SCIPgetRootDialog(scip);
   if( root == NULL )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &root, SCIPdialogExecMenuLazy, NULL,
            "SCIP", "SCIP's main menu", TRUE, NULL) );
      SCIP_CALL( SCIPsetRootDialog(scip, root) );
      SCIP_CALL( SCIPreleaseDialog(scip, &root) );
      root = SCIPgetRootDialog(scip);
   }

   /* checksol */
   if( !SCIPdialogHasEntry(root, "checksol") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecChecksol, NULL,
            "checksol", "double checks best solution w.r.t. original problem", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* conflictgraph */
   if( !SCIPdialogHasEntry(root, "conflictgraph") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecConflictgraph, NULL,
            "conflictgraph", "writes binary variable implications of transformed problem as conflict graph to file",
            FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* display */
   if( !SCIPdialogHasEntry(root, "display") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "display", "display information", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(root, "display", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;
   
   /* display branching */
   if( !SCIPdialogHasEntry(submenu, "branching") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayBranching, NULL,
            "branching", "display branching rules", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display conflict */
   if( !SCIPdialogHasEntry(submenu, "conflict") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayConflict, NULL,
            "conflict", "display conflict handlers", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display conshdlrs */
   if( !SCIPdialogHasEntry(submenu, "conshdlrs") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayConshdlrs, NULL,
            "conshdlrs", "display constraint handlers", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display displaycols */
   if( !SCIPdialogHasEntry(submenu, "displaycols") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayDisplaycols, NULL,
            "displaycols", "display display columns", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display heuristics */
   if( !SCIPdialogHasEntry(submenu, "heuristics") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayHeuristics, NULL,
            "heuristics", "display primal heuristics", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display memory */
   if( !SCIPdialogHasEntry(submenu, "memory") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayMemory, NULL,
            "memory", "display memory diagnostics", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* display nodeselectors */
   if( !SCIPdialogHasEntry(submenu, "nodeselectors") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayNodeselectors, NULL,
            "nodeselectors", "display node selectors", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display presolvers */
   if( !SCIPdialogHasEntry(submenu, "presolvers") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayPresolvers, NULL,
            "presolvers", "display presolvers", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display problem */
   if( !SCIPdialogHasEntry(submenu, "problem") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayProblem, NULL,
            "problem", "display original problem", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display propagators */
   if( !SCIPdialogHasEntry(submenu, "propagators") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayPropagators, NULL,
            "propagators", "display propagators", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* display readers */
   if( !SCIPdialogHasEntry(submenu, "readers") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayReaders, NULL,
            "readers", "display file readers", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display separators */
   if( !SCIPdialogHasEntry(submenu, "separators") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplaySeparators, NULL,
            "separators", "display cut separators", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display solution */
   if( !SCIPdialogHasEntry(submenu, "solution") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplaySolution, NULL,
            "solution", "display best primal solution", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display statistics */
   if( !SCIPdialogHasEntry(submenu, "statistics") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayStatistics, NULL,
            "statistics", "display problem and optimization statistics", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display transproblem */
   if( !SCIPdialogHasEntry(submenu, "transproblem") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayTransproblem, NULL,
            "transproblem", "display transformed/preprocessed problem", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display value */
   if( !SCIPdialogHasEntry(submenu, "value") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayValue, NULL,
            "value", "display value of single variable in best primal solution", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display varbranchstatistics */
   if( !SCIPdialogHasEntry(submenu, "varbranchstatistics") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayVarbranchstatistics, NULL,
            "varbranchstatistics", "display statistics for branching on variables", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* display transsolution */
   if( !SCIPdialogHasEntry(submenu, "transsolution") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecDisplayTranssolution, NULL,
            "transsolution", "display best primal solution in transformed variables", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* free */
   if( !SCIPdialogHasEntry(root, "free") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecFree, NULL,
            "free", "free current problem from memory", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* help */
   if( !SCIPdialogHasEntry(root, "help") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecHelp, NULL,
            "help", "display this help", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* newstart */
   if( !SCIPdialogHasEntry(root, "newstart") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecNewstart, NULL,
            "newstart", "reset branch and bound tree to start again from root", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* optimize */
   if( !SCIPdialogHasEntry(root, "optimize") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecOptimize, NULL,
            "optimize", "solve the problem", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* presolve */
   if( !SCIPdialogHasEntry(root, "presolve") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecPresolve, NULL,
            "presolve", "solve the problem, but stop after presolving stage", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* quit */
   if( !SCIPdialogHasEntry(root, "quit") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecQuit, NULL,
            "quit", "leave SCIP", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* read */
   if( !SCIPdialogHasEntry(root, "read") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecRead, NULL,
            "read", "read a problem", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* set */
   SCIP_CALL( SCIPincludeDialogDefaultSet(scip) );

   /* write */
   if( !SCIPdialogHasEntry(root, "write") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "write", "write information to file", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(root, "write", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;
   
   /* write problem */
   if( !SCIPdialogHasEntry(submenu, "problem") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecWriteProblem, NULL,
            "problem", "write original problem in CIP format to file", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* write solution */
   if( !SCIPdialogHasEntry(submenu, "solution") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecWriteSolution, NULL,
            "solution", "write best primal solution to file", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   /* write statistics */
   if( !SCIPdialogHasEntry(submenu, "statistics") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecWriteStatistics, NULL,
            "statistics", "write statistics to file", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* write transproblem */
   if( !SCIPdialogHasEntry(submenu, "transproblem") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecWriteTransproblem, NULL,
            "transproblem", "write transformed (preprocessed) problem in CIP format to file", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }
   
   return SCIP_OKAY;
}

/** if a '/' occurs in the parameter's name, adds a sub menu dialog to the given menu and inserts the parameter dialog
 *  recursively in the sub menu; if no '/' occurs in the name, adds a parameter change dialog into the given dialog menu
 */
static
SCIP_RETCODE addParamDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG*          menu,               /**< dialog menu to insert the parameter into */
   SCIP_PARAM*           param,              /**< parameter to add a dialog for */
   char*                 paramname           /**< parameter name to parse */
   )
{
   char* slash;
   char* dirname;

   assert(paramname != NULL);

   /* check for a '/' */
   slash = strchr(paramname, '/');

   if( slash == NULL )
   {
      /* check, if the corresponding dialog already exists */
      if( !SCIPdialogHasEntry(menu, paramname) )
      {
         SCIP_DIALOG* paramdialog;

         /* create a parameter change dialog */
         SCIP_CALL( SCIPcreateDialog(scip, &paramdialog, SCIPdialogExecSetParam, SCIPdialogDescSetParam, 
               paramname, SCIPparamGetDesc(param), FALSE, (SCIP_DIALOGDATA*)param) );
         SCIP_CALL( SCIPaddDialogEntry(scip, menu, paramdialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &paramdialog) );
      }
   }
   else
   {
      SCIP_DIALOG* submenu;

      /* split the parameter name into dirname and parameter name */
      dirname = paramname;
      paramname = slash+1;
      *slash = '\0';

      /* if not yet existing, create a corresponding sub menu */
      if( !SCIPdialogHasEntry(menu, dirname) )
      {
         char desc[SCIP_MAXSTRLEN];

         sprintf(desc, "parameters for <%s>", dirname);
         SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL, dirname, desc, TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, menu, submenu) );
         SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
      }

      /* find the corresponding sub menu */
      (void)SCIPdialogFindEntry(menu, dirname, &submenu);
      if( submenu == NULL )
      {
         SCIPerrorMessage("dialog sub menu not found\n");
         return SCIP_PLUGINNOTFOUND;
      }

      /* recursively call add parameter method */
      SCIP_CALL( addParamDialog(scip, submenu, param, paramname) );
   }

   return SCIP_OKAY;
}

/** includes or updates the "set" menu for each available parameter setting */
SCIP_RETCODE SCIPincludeDialogDefaultSet(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_DIALOG* root;
   SCIP_DIALOG* setmenu;
   SCIP_DIALOG* submenu;
   SCIP_DIALOG* dialog;
   SCIP_PARAM** params;
   const char* pname;
   char* paramname;
   int nparams;
   int i;

   /* get root dialog */
   root = SCIPgetRootDialog(scip);
   if( root == NULL )
   {
      SCIPerrorMessage("root dialog not found\n");
      return SCIP_PLUGINNOTFOUND;
   }

   /* find (or create) the "set" menu of the root dialog */
   if( !SCIPdialogHasEntry(root, "set") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &setmenu, SCIPdialogExecMenu, NULL,
            "set", "load/save/change parameters", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, root, setmenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &setmenu) );
   }
   if( SCIPdialogFindEntry(root, "set", &setmenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   /* set load */
   if( !SCIPdialogHasEntry(setmenu, "load") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecSetLoad, NULL,
            "load", "load parameter settings from a file", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* set save */
   if( !SCIPdialogHasEntry(setmenu, "save") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecSetSave, NULL,
            "save", "save parameter settings to a file", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* set diffsave */
   if( !SCIPdialogHasEntry(setmenu, "diffsave") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecSetDiffsave, NULL,
            "diffsave", "save non-default parameter settings to a file", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* set branching */
   if( !SCIPdialogHasEntry(setmenu, "branching") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "branching", "change parameters for branching rules", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "branching", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNBranchrules(scip); ++i )
   {
      SCIP_BRANCHRULE* branchrule = SCIPgetBranchrules(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPbranchruleGetName(branchrule)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPbranchruleGetName(branchrule), SCIPbranchruleGetDesc(branchrule), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set branching priority */
   if( !SCIPdialogHasEntry(submenu, "priority") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecSetBranchingPriority, NULL,
            "priority", "change branching priority of a single variable", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* set branching direction */
   if( !SCIPdialogHasEntry(submenu, "direction") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecSetBranchingDirection, NULL,
            "direction", "change preferred branching direction of a single variable (-1:down, 0:auto, +1:up)",
            FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   /* set conflict */
   if( !SCIPdialogHasEntry(setmenu, "conflict") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "conflict", "change parameters for conflict handlers", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "conflict", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNConflicthdlrs(scip); ++i )
   {
      SCIP_CONFLICTHDLR* conflicthdlr = SCIPgetConflicthdlrs(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPconflicthdlrGetName(conflicthdlr)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPconflicthdlrGetName(conflicthdlr), SCIPconflicthdlrGetDesc(conflicthdlr), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set constraints */
   if( !SCIPdialogHasEntry(setmenu, "constraints") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "constraints", "change parameters for constraint handlers", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "constraints", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNConshdlrs(scip); ++i )
   {
      SCIP_CONSHDLR* conshdlr = SCIPgetConshdlrs(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPconshdlrGetName(conshdlr)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPconshdlrGetName(conshdlr), SCIPconshdlrGetDesc(conshdlr), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set display */
   if( !SCIPdialogHasEntry(setmenu, "display") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "display", "change parameters for display columns", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "display", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNDisps(scip); ++i )
   {
      SCIP_DISP* disp = SCIPgetDisps(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPdispGetName(disp)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPdispGetName(disp), SCIPdispGetDesc(disp), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set heuristics */
   if( !SCIPdialogHasEntry(setmenu, "heuristics") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "heuristics", "change parameters for primal heuristics", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "heuristics", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNHeurs(scip); ++i )
   {
      SCIP_HEUR* heur = SCIPgetHeurs(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPheurGetName(heur)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPheurGetName(heur), SCIPheurGetDesc(heur), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set limits */
   if( !SCIPdialogHasEntry(setmenu, "limits") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "limits", "change parameters for time, memory, objective value, and other limits", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );

      SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecSetLimitsObjective, NULL,
            "objective", "set limit on objective value", FALSE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }

   /* set lp */
   if( !SCIPdialogHasEntry(setmenu, "lp") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "lp", "change parameters for linear programming relaxations", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }

   /* set memory */
   if( !SCIPdialogHasEntry(setmenu, "memory") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "memory", "change parameters for memory management", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }

   /* set misc */
   if( !SCIPdialogHasEntry(setmenu, "misc") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "misc", "change parameters for miscellaneous stuff", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }

   /* set nodeselection */
   if( !SCIPdialogHasEntry(setmenu, "nodeselection") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "nodeselection", "change parameters for node selectors", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "nodeselection", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNNodesels(scip); ++i )
   {
      SCIP_NODESEL* nodesel = SCIPgetNodesels(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPnodeselGetName(nodesel)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPnodeselGetName(nodesel), SCIPnodeselGetDesc(nodesel), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set numerics */
   if( !SCIPdialogHasEntry(setmenu, "numerics") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "numerics", "change parameters for numerical values", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }

   /* set presolving */
   if( !SCIPdialogHasEntry(setmenu, "presolving") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "presolving", "change parameters for presolving", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "presolving", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNPresols(scip); ++i )
   {
      SCIP_PRESOL* presol = SCIPgetPresols(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPpresolGetName(presol)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPpresolGetName(presol), SCIPpresolGetDesc(presol), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set pricing */
   if( !SCIPdialogHasEntry(setmenu, "pricing") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "pricing", "change parameters for pricing variables", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "pricing", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNPricers(scip); ++i )
   {
      SCIP_PRICER* pricer = SCIPgetPricers(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPpricerGetName(pricer)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPpricerGetName(pricer), SCIPpricerGetDesc(pricer), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set propagation */
   if( !SCIPdialogHasEntry(setmenu, "propagating") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "propagating", "change parameters for constraint propagation", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }

   /* set reading */
   if( !SCIPdialogHasEntry(setmenu, "reading") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "reading", "change parameters for problem file readers", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "reading", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNReaders(scip); ++i )
   {
      SCIP_READER* reader = SCIPgetReaders(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPreaderGetName(reader)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPreaderGetName(reader), SCIPreaderGetDesc(reader), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set separating */
   if( !SCIPdialogHasEntry(setmenu, "separating") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "separating", "change parameters for cut separators", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }
   if( SCIPdialogFindEntry(setmenu, "separating", &submenu) != 1 )
      return SCIP_PLUGINNOTFOUND;

   for( i = 0; i < SCIPgetNSepas(scip); ++i )
   {
      SCIP_SEPA* sepa = SCIPgetSepas(scip)[i];

      if( !SCIPdialogHasEntry(submenu, SCIPsepaGetName(sepa)) )
      {
         SCIP_CALL( SCIPcreateDialog(scip, &dialog, SCIPdialogExecMenu, NULL,
               SCIPsepaGetName(sepa), SCIPsepaGetDesc(sepa), TRUE, NULL) );
         SCIP_CALL( SCIPaddDialogEntry(scip, submenu, dialog) );
         SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
      }
   }

   /* set timing */
   if( !SCIPdialogHasEntry(setmenu, "timing") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "timing", "change parameters for timing issues", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }

   /* set vbc */
   if( !SCIPdialogHasEntry(setmenu, "vbc") )
   {
      SCIP_CALL( SCIPcreateDialog(scip, &submenu, SCIPdialogExecMenu, NULL,
            "vbc", "change parameters for VBC tool output", TRUE, NULL) );
      SCIP_CALL( SCIPaddDialogEntry(scip, setmenu, submenu) );
      SCIP_CALL( SCIPreleaseDialog(scip, &submenu) );
   }

   /* get SCIP's parameters */
   params = SCIPgetParams(scip);
   nparams = SCIPgetNParams(scip);

   /* insert each parameter into the set menu */
   for( i = 0; i < nparams; ++i )
   {
      pname = SCIPparamGetName(params[i]);
      SCIP_ALLOC( BMSduplicateMemoryArray(&paramname, pname, strlen(pname)+1) );
      SCIP_CALL( addParamDialog(scip, setmenu, params[i], paramname) );
      BMSfreeMemoryArray(&paramname);
   }

   return SCIP_OKAY;
}

