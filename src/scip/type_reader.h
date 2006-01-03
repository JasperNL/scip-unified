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
#pragma ident "@(#) $Id: type_reader.h,v 1.12 2006/01/03 12:22:59 bzfpfend Exp $"

/**@file   type_reader.h
 * @brief  type definitions for input file readers
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_TYPE_READER_H__
#define __SCIP_TYPE_READER_H__


typedef struct SCIP_Reader SCIP_READER;               /**< reader data structure */
typedef struct SCIP_ReaderData SCIP_READERDATA;       /**< reader specific data */


/** destructor of reader to free user data (called when SCIP is exiting)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - reader          : the reader itself
 */
#define SCIP_DECL_READERFREE(x) SCIP_RETCODE x (SCIP* scip, SCIP_READER* reader)

/** problem reading method of reader
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - reader          : the reader itself
 *  - filename        : full path and name of file to read, or NULL if stdin should be used
 *  - result          : pointer to store the result of the file reading call
 *
 *  possible return values for *result:
 *  - SCIP_SUCCESS    : the reader read the file correctly and created an appropritate problem
 *  - SCIP_DIDNOTRUN  : the reader is not responsible for given input file
 *
 *  If the reader detected an error in the input file, it should return with RETCODE SCIP_READERROR or SCIP_NOFILE.
 */
#define SCIP_DECL_READERREAD(x) SCIP_RETCODE x (SCIP* scip, SCIP_READER* reader, const char* filename, SCIP_RESULT* result)



#include "scip/def.h"
#include "scip/type_retcode.h"
#include "scip/type_result.h"
#include "scip/type_scip.h"


#endif
