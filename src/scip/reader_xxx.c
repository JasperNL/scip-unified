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
#pragma ident "@(#) $Id: reader_xxx.c,v 1.15 2006/01/03 12:22:53 bzfpfend Exp $"

/**@file   reader_xxx.c
 * @brief  XXX file reader
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "scip/reader_xxx.h"


#define READER_NAME             "xxxreader"
#define READER_DESC             "xxx file reader"
#define READER_EXTENSION        "xxx"




/*
 * Data structures
 */

/* TODO: (optional) fill in the necessary reader data */

/** data for xxx reader */
struct SCIP_ReaderData
{
};




/*
 * Local methods
 */

/* put your local methods here, and declare them static */





/*
 * Callback methods of reader
 */

/** destructor of reader to free user data (called when SCIP is exiting) */
#if 0
static
SCIP_DECL_READERFREE(readerFreeXxx)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xxx reader not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define readerFreeXxx NULL
#endif


/** problem reading method of reader */
static
SCIP_DECL_READERREAD(readerReadXxx)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xxx reader not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}




/*
 * reader specific interface methods
 */

/** includes the xxx file reader in SCIP */
SCIP_RETCODE SCIPincludeReaderXxx(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_READERDATA* readerdata;

   /* create xxx reader data */
   readerdata = NULL;
   /* TODO: (optional) create reader specific data here */
   
   /* include xxx reader */
   SCIP_CALL( SCIPincludeReader(scip, READER_NAME, READER_DESC, READER_EXTENSION,
         readerFreeXxx, readerReadXxx, readerdata) );

   /* add xxx reader parameters */
   /* TODO: (optional) add reader specific parameters with SCIPaddTypeParam() here */
   
   return SCIP_OKAY;
}
