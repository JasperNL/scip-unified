/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2009 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: reader_cip.c,v 1.2.2.1 2009/06/19 07:53:48 bzfwolte Exp $"

/**@file   reader_cip.c
 * @ingroup FILEREADERS 
 * @brief  CIP file reader
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "scip/reader_cip.h"


#define READER_NAME             "cipreader"
#define READER_DESC             "file reader for CIP (Constraint Integer Program) format"
#define READER_EXTENSION        "cip"




/*
 * Data structures
 */

/** data for cip reader */
struct SCIP_ReaderData
{
	char dummy; /* to have at least one member */
};




/*
 * Local methods
 */

/* put your local methods here, and declare them static */





/*
 * Callback methods of reader
 */

/** destructor of reader to free user data (called when SCIP is exiting) */
#define readerFreeCip NULL


/** problem reading method of reader */
#define readerReadCip NULL


/** problem writing method of reader */
static
SCIP_DECL_READERWRITE(readerWriteCip)
{  /*lint --e{715}*/

   int i;

   SCIPinfoMessage(scip, file, "STATISTICS\n");
   SCIPinfoMessage(scip, file, "  Problem name     : %s\n", name);
   SCIPinfoMessage(scip, file, "  Variables        : %d (%d binary, %d integer, %d implicit integer, %d continuous)\n",
      nvars, nbinvars, nintvars, nimplvars, ncontvars);
   SCIPinfoMessage(scip, file, "  Constraints      : %d initial, %d maximal\n", startnconss, maxnconss);
   
   SCIPinfoMessage(scip, file, "OBJECTIVE\n");
   SCIPinfoMessage(scip, file, "  Sense            : %s\n", objsense == SCIP_OBJSENSE_MINIMIZE ? "minimize" : "maximize");
   if( !SCIPisZero(scip, objoffset) )
      SCIPinfoMessage(scip, file, "  Offset           : %+.15g\n", objoffset);
   if( !SCIPisEQ(scip, objscale, 1.0) )
      SCIPinfoMessage(scip, file, "  Scale            : %.15g\n", objscale);

   if( nvars > 0 )
   {
      SCIPinfoMessage(scip, file, "VARIABLES\n");
      for( i = 0; i < nvars; ++i )
         SCIP_CALL( SCIPprintVar(scip, vars[i], file) );
   }

   if( nfixedvars > 0 )
   {
      SCIPinfoMessage(scip, file, "FIXED\n");
      for( i = 0; i < nfixedvars; ++i )
         SCIP_CALL( SCIPprintVar(scip, fixedvars[i], file) );
   }

   if( nconss > 0 )
   {
      SCIPinfoMessage(scip, file, "CONSTRAINTS\n");
      
      for( i = 0; i < nconss; ++i )
      {
         /* in case the transformed is written only constraint are posted which are enabled in the current node */
         if( transformed && !SCIPconsIsEnabled(conss[i]) )
            continue;
         
         SCIP_CALL( SCIPprintCons(scip, conss[i], file) );
      }
   }
   
   *result = SCIP_SUCCESS;

   SCIPinfoMessage(scip, file, "END\n");
   return SCIP_OKAY;
}


/*
 * reader specific interface methods
 */

/** includes the cip file reader in SCIP */
SCIP_RETCODE SCIPincludeReaderCip(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_READERDATA* readerdata;

   /* create cip reader data */
   readerdata = NULL;
   /* TODO: (optional) create reader specific data here */
   
   /* include cip reader */
   SCIP_CALL( SCIPincludeReader(scip, READER_NAME, READER_DESC, READER_EXTENSION,
         readerFreeCip, readerReadCip, readerWriteCip, readerdata) );

   /* add cip reader parameters */
   /* TODO: (optional) add reader specific parameters with SCIPaddTypeParam() here */
   
   return SCIP_OKAY;
}
