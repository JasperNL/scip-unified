/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2015 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   dialog_xyz.c
 * @brief  xyz user interface dialog
 * @author Kati Wolter
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "scip/dialog_xyz.h"


#define DIALOG_NAME            "xyz"
#define DIALOG_DESC            "xyz user interface dialog"
#define DIALOG_ISSUBMENU          FALSE 




/*
 * Data structures
 */

/* TODO: fill in the necessary dialog data */

/** dialog data */
struct SCIP_DialogData
{
};




/*
 * Local methods
 */

/* put your local methods here, and declare them static */




/*
 * Callback methods of dialog
 */

/* TODO: Implement all necessary dialog methods. The methods with an #if 0 ... #else #define ... are optional */


/** copy method for dialog plugins (called when SCIP copies plugins) */
#if 0
static
SCIP_DECL_DIALOGCOPY(dialogCopyXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz dialog not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define dialogCopyXyz NULL
#endif

/** destructor of dialog to free user data (called when the dialog is not captured anymore) */
#if 0
static
SCIP_DECL_DIALOGFREE(dialogFreeXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz dialog not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define dialogFreeXyz NULL
#endif

/** description output method of dialog */
#if 0
static
SCIP_DECL_DIALOGDESC(dialogDescXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz dialog not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define dialogDescXyz NULL
#endif


/** execution method of dialog */
static
SCIP_DECL_DIALOGEXEC(dialogExecXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz dialog not implemented yet\n");
   SCIPABORT(); /*lint --e{827}*/

   /* add your dialog to history of dialogs that have been executed */
   SCIP_CALL( SCIPdialoghdlrAddHistory(dialoghdlr, dialog, NULL, FALSE) );

   /* TODO: Implement execution of your dialog here. */

   /* next dialog will be root dialog again */
   *nextdialog = SCIPdialoghdlrGetRoot(dialoghdlr);

   return SCIP_OKAY;
}





/*
 * dialog specific interface methods
 */

/** creates the xyz dialog and includes it in SCIP */
SCIP_RETCODE SCIPincludeDialogXyz(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_DIALOGDATA* dialogdata;
   SCIP_DIALOG* dialog;
   SCIP_DIALOG* parentdialog;

   /* create xyz dialog data */
   dialogdata = NULL;
   /* TODO: (optional) create dialog specific data here */

   /* get parent dialog */
   parentdialog = SCIPgetRootDialog(scip);
   assert(parentdialog != NULL);
   /* TODO: (optional) change parent dialog from root dialog to another existing dialog (needs to be a menu) */

   /* create, include, and release dialog */
   if( !SCIPdialogHasEntry(parentdialog, DIALOG_NAME) )
   {
      SCIP_CALL( SCIPincludeDialog(scip, &dialog, 
            dialogCopyXyz, dialogExecXyz, dialogDescXyz, dialogFreeXyz,
            DIALOG_NAME, DIALOG_DESC, DIALOG_ISSUBMENU, dialogdata) );
      SCIP_CALL( SCIPaddDialogEntry(scip, parentdialog, dialog) );
      SCIP_CALL( SCIPreleaseDialog(scip, &dialog) );
   }

   return SCIP_OKAY;
}
