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
#pragma ident "@(#) $Id: reader_mps.c,v 1.63 2006/01/03 12:22:53 bzfpfend Exp $"

/**@file   reader_mps.c
 * @brief  MPS file reader
 * @author Thorsten Koch
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>
#include <ctype.h>

#include "scip/reader_mps.h"
#include "scip/cons_linear.h"


#define READER_NAME             "mpsreader"
#define READER_DESC             "file reader for MIPs in IBM's Mathematical Programming System format"
#define READER_EXTENSION        "mps"



/*
 * mps reader internal methods
 */

#define MPS_MAX_LINELEN 256

#define PATCH_CHAR    '_'
#define BLANK         ' '

enum MpsSection
{
   MPS_NAME, MPS_OBJSEN, MPS_OBJNAME, MPS_ROWS, MPS_COLUMNS, MPS_RHS, MPS_RANGES, MPS_BOUNDS, MPS_ENDATA
};
typedef enum MpsSection MPSSECTION;

struct MpsInput
{
   MPSSECTION           section;
   SCIP_FILE*           fp;
   int                  lineno;
   SCIP_OBJSENSE        objsense;
   SCIP_Bool            haserror;
   char                 buf[MPS_MAX_LINELEN];
   const char*          f0;
   const char*          f1;
   const char*          f2;
   const char*          f3;
   const char*          f4;
   const char*          f5;
   char                 probname[MPS_MAX_LINELEN];
   char                 objname [MPS_MAX_LINELEN];
   SCIP_Bool            isinteger;
   SCIP_Bool            isnewformat;
};
typedef struct MpsInput MPSINPUT;



static
SCIP_RETCODE mpsinputCreate(
   SCIP*                 scip,
   MPSINPUT**            mpsi,
   SCIP_FILE*            fp
   )
{
   assert(mpsi != NULL);
   assert(fp != NULL);

   SCIP_CALL( SCIPallocMemory(scip, mpsi) );

   (*mpsi)->section     = MPS_NAME;
   (*mpsi)->fp          = fp;
   (*mpsi)->lineno      = 0;
   (*mpsi)->objsense    = SCIP_OBJSENSE_MINIMIZE;
   (*mpsi)->haserror    = FALSE;
   (*mpsi)->isinteger   = FALSE;
   (*mpsi)->isnewformat = FALSE;
   (*mpsi)->buf     [0] = '\0';
   (*mpsi)->probname[0] = '\0';
   (*mpsi)->objname [0] = '\0';
   (*mpsi)->f0          = NULL;
   (*mpsi)->f1          = NULL;
   (*mpsi)->f2          = NULL;
   (*mpsi)->f3          = NULL;
   (*mpsi)->f4          = NULL;
   (*mpsi)->f5          = NULL;

   return SCIP_OKAY;
}

static
void mpsinputFree(
   SCIP*                 scip,
   MPSINPUT**            mpsi
   )
{
   SCIPfreeMemory(scip, mpsi);
}

static
MPSSECTION mpsinputSection(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->section;
}

#if 0
static
int mpsinputLineno(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->lineno;
}
#endif

static
const char* mpsinputField0(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f0;
}

static
const char* mpsinputField1(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f1;
}

static
const char* mpsinputField2(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f2;
}

static
const char* mpsinputField3(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f3;
}

static
const char* mpsinputField4(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f4;
}

static
const char* mpsinputField5(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f5;
}

#if 0
static
const char* mpsinputProbname(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->probname;
}
#endif

static
const char* mpsinputObjname(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->objname;
}

static
SCIP_OBJSENSE mpsinputObjsense(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->objsense;
}

static
SCIP_Bool mpsinputHasError(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->haserror;
}

static
SCIP_Bool mpsinputIsInteger(
   const MPSINPUT*       mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->isinteger;
}

static
void mpsinputSetSection(
   MPSINPUT*             mpsi,
   MPSSECTION            section
   )
{
   assert(mpsi != NULL);

   mpsi->section = section;
}

static
void mpsinputSetProbname(
   MPSINPUT*             mpsi,
   const char*           probname
   )
{
   assert(mpsi     != NULL);
   assert(probname != NULL);
   assert(strlen(probname) < sizeof(mpsi->probname));
   
   strcpy(mpsi->probname, probname);
}

static
void mpsinputSetObjname(
   MPSINPUT*             mpsi, 
   const char*           objname
   )
{
   assert(mpsi != NULL);
   assert(objname != NULL);
   assert(strlen(objname) < sizeof(mpsi->objname));

   strcpy(mpsi->objname, objname);
}

static
void mpsinputSetObjsense(
   MPSINPUT*             mpsi,
   SCIP_OBJSENSE         sense
   )
{
   assert(mpsi != NULL);

   mpsi->objsense = sense;
}

static
void mpsinputSyntaxerror(
   MPSINPUT*             mpsi
   ) 
{
   assert(mpsi != NULL);

   SCIPwarningMessage("Syntax error in line %d\n", mpsi->lineno);
   mpsi->section  = MPS_ENDATA;
   mpsi->haserror = TRUE;
}

static
void mpsinputEntryIgnored(
   SCIP*                 scip,               /**< SCIP data structure */
   MPSINPUT*             mpsi, 
   const char*           what, 
   const char*           what_name, 
   const char*           entity, 
   const char*           entity_name
   )
{
   assert(mpsi        != NULL);
   assert(what        != NULL);
   assert(what_name   != NULL);
   assert(entity      != NULL);
   assert(entity_name != NULL);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL,
      "Warning line %d: %s \"%s\" for %s \"%s\" ignored\n", mpsi->lineno, what, what_name, entity, entity_name);
}

/* fill the line from \p pos up to column 80 with blanks.
 */
static
void clearFrom(
   char*                 buf,
   unsigned int          pos
   )
{
   unsigned int i;

   for(i = pos; i < 80; i++)
      buf[i] = BLANK;
   buf[80] = '\0';
}

/* change all blanks inside a field to #PATCH_CHAR.
 */
static
void patchField(
   char*                 buf,
   int                   beg,
   int                   end
   )
{
   int i;

   while((beg <= end) && (buf[end] == BLANK))
      end--;

   while((beg <= end) && (buf[beg] == BLANK))
      beg++;

   for(i = beg; i <= end; i++)
      if (buf[i] == BLANK)
         buf[i] = PATCH_CHAR;
}

/* read a mps format data line and parse the fields.
 */
static
SCIP_Bool mpsinputReadLine(
   MPSINPUT*             mpsi
   )
{
   unsigned int len;
   unsigned int i;
   int space;
   char* s;
   SCIP_Bool is_marker;
   char* nexttok;

   do
   {
      mpsi->f0  = mpsi->f1 = mpsi->f2 = mpsi->f3 = mpsi->f4 = mpsi->f5 = 0;
      is_marker = FALSE;
   
      /* Read until we have a not comment line.
       */
      do
      {
         if (NULL == SCIPfgets(mpsi->buf, sizeof(mpsi->buf), mpsi->fp))
            return FALSE;
         mpsi->lineno++;
      } 
      while(*mpsi->buf == '*');

      /* Normalize line
       */
      len = strlen(mpsi->buf);

      for(i = 0; i < len; i++)
         if ((mpsi->buf[i] == '\t') || (mpsi->buf[i] == '\n') || (mpsi->buf[i] == '\r'))
            mpsi->buf[i] = BLANK;
      
      if (len < 80)
         clearFrom(mpsi->buf, len);

      assert(strlen(mpsi->buf) >= 80);

      /* Look for new section
       */
      if (*mpsi->buf != BLANK)
      {
         mpsi->f0 = SCIPstrtok(&mpsi->buf[0], " ", &nexttok);

         assert(mpsi->f0 != 0);

         mpsi->f1 = SCIPstrtok(NULL, " ", &nexttok);

         return TRUE;
      }

      /* If we decide to use the new format we never revert thsi decision
       */
      if (!mpsi->isnewformat)
      {
         /* Test for fixed format comments
          */
         if ((mpsi->buf[14] == '$') && (mpsi->buf[13] == ' '))
            clearFrom(mpsi->buf, 14);
         else if ((mpsi->buf[39] == '$') && (mpsi->buf[38] == ' '))
            clearFrom(mpsi->buf, 39);

         /* Test for fixed format
          */
         space = mpsi->buf[12] | mpsi->buf[13] 
            | mpsi->buf[22] | mpsi->buf[23] 
            | mpsi->buf[36] | mpsi->buf[37] | mpsi->buf[38]
            | mpsi->buf[47] | mpsi->buf[48] 
            | mpsi->buf[61] | mpsi->buf[62] | mpsi->buf[63];

         if (space == BLANK)
         {
            /* Now we have space at the right positions.
             * But are there also the non space where they
             * should be ?
             */
            SCIP_Bool number = isdigit(mpsi->buf[24]) || isdigit(mpsi->buf[25]) 
               || isdigit(mpsi->buf[26]) || isdigit(mpsi->buf[27]) 
               || isdigit(mpsi->buf[28]) || isdigit(mpsi->buf[29]) 
               || isdigit(mpsi->buf[30]) || isdigit(mpsi->buf[31]) 
               || isdigit(mpsi->buf[32]) || isdigit(mpsi->buf[33]) 
               || isdigit(mpsi->buf[34]) || isdigit(mpsi->buf[35]); 
            
            /* len < 13 is handle ROW lines with embedded spaces
             * in the names correctly
             */
            if (number || len < 13)
            {
               /* We assume fixed format, so we patch possible embedded spaces.
                */
               patchField(mpsi->buf,  4, 12);
               patchField(mpsi->buf, 14, 22);
               patchField(mpsi->buf, 39, 47);
            }
            else
            {
               if (  mpsi->section == MPS_COLUMNS || mpsi->section == MPS_RHS
                  || mpsi->section == MPS_RANGES  || mpsi->section == MPS_BOUNDS)
                  mpsi->isnewformat = TRUE;
            }
         }
         else
         {
            mpsi->isnewformat = TRUE;
         }
      }
      s = &mpsi->buf[1];
      
      /* At this point it is not clear if we have a indicator field.
       * If there is none (e.g. empty) f1 will be the first name field.
       * If there is one, f2 will be the first name field.
       * 
       * Initially comment marks '$' ar only allowed in the beginning
       * of the 2nd and 3rd name field. We test all fields but the first.
       * This makes no difference, since if the $ is at the start of a value
       * field, the line will be errornous anyway.
       */
      do
      {
         if (NULL == (mpsi->f1 = SCIPstrtok(s, " ", &nexttok)))
            break;
         
         if ((NULL == (mpsi->f2 = SCIPstrtok(NULL, " ", &nexttok))) || (*mpsi->f2 == '$'))
         {
            mpsi->f2 = 0;
            break;      
         }
         if (!strcmp(mpsi->f2, "'MARKER'"))
            is_marker = TRUE;
            
         if ((NULL == (mpsi->f3 = SCIPstrtok(NULL, " ", &nexttok))) || (*mpsi->f3 == '$'))
         {
            mpsi->f3 = 0;
            break;      
         }
         if (is_marker)
         {
            if (!strcmp(mpsi->f3, "'INTORG'"))
               mpsi->isinteger = TRUE;
            else if (!strcmp(mpsi->f3, "'INTEND'"))
               mpsi->isinteger = FALSE;
            else
               break; /* unknown marker */
         }
         if (!strcmp(mpsi->f3, "'MARKER'"))
            is_marker = TRUE;

         if ((NULL == (mpsi->f4 = SCIPstrtok(NULL, " ", &nexttok))) || (*mpsi->f4 == '$'))
         {
            mpsi->f4 = 0;
            break;      
         }
         if (is_marker)
         {
            if (!strcmp(mpsi->f4, "'INTORG'"))
               mpsi->isinteger = TRUE;
            else if (!strcmp(mpsi->f4, "'INTEND'"))
               mpsi->isinteger = FALSE;
            else
               break; /* unknown marker */
         }
         if ((NULL == (mpsi->f5 = SCIPstrtok(NULL, " ", &nexttok))) || (*mpsi->f5 == '$'))
            mpsi->f5 = 0;
      }
      while(FALSE);
   }
   while(is_marker);

   return TRUE;
}

/* Insert \p name as field 1 and shift all other fields up.
 */
static
void mpsinputInsertName(
   MPSINPUT*             mpsi,
   const char*           name,
   SCIP_Bool             second
   )
{
   assert(mpsi != NULL);
   assert(name != NULL);

   mpsi->f5 = mpsi->f4;
   mpsi->f4 = mpsi->f3;
   mpsi->f3 = mpsi->f2;

   if (second)
      mpsi->f2 = name;
   else
   {
      mpsi->f2 = mpsi->f1;
      mpsi->f1 = name;
   }
}

/* Process NAME section.
 */
static
SCIP_RETCODE readName(
   MPSINPUT*             mpsi
   )
{
   assert(mpsi != NULL);

   /* This has to be the Line with the NAME section.
    */
   if (!mpsinputReadLine(mpsi) || mpsinputField0(mpsi) == NULL || strcmp(mpsinputField0(mpsi), "NAME"))
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }

   /* Sometimes the name is omitted. */
   mpsinputSetProbname(mpsi, (mpsinputField1(mpsi) == 0) ? "_MPS_" : mpsinputField1(mpsi));
   
   /*printf("Problem name   : %s\n", mpsinputProbname(mpsi));*/
   
   /* This hat to be a new section */
   if (!mpsinputReadLine(mpsi) || (mpsinputField0(mpsi) == NULL))
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }

   if (!strcmp(mpsinputField0(mpsi), "ROWS"))
      mpsinputSetSection(mpsi, MPS_ROWS);
   else if (!strcmp(mpsinputField0(mpsi), "OBJSEN"))
      mpsinputSetSection(mpsi, MPS_OBJSEN);
   else if (!strcmp(mpsinputField0(mpsi), "OBJNAME"))
      mpsinputSetSection(mpsi, MPS_OBJNAME);
   else
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }

   return SCIP_OKAY;
}

/* Process OBJSEN section. This Section is an ILOG extension.
 */
static
SCIP_RETCODE readObjsen(
   MPSINPUT*             mpsi
   )
{
   assert(mpsi != NULL);

   /* This has to be the Line with MIN or MAX. */
   if (!mpsinputReadLine(mpsi) || (mpsinputField1(mpsi) == NULL))
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }

   if (strcmp(mpsinputField1(mpsi), "MIN"))
      mpsinputSetObjsense(mpsi, SCIP_OBJSENSE_MINIMIZE);
   else if (strcmp(mpsinputField1(mpsi), "MAX"))
      mpsinputSetObjsense(mpsi, SCIP_OBJSENSE_MAXIMIZE);
   else
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }

   /* Look for ROWS or OBJNAME Section */
   if (!mpsinputReadLine(mpsi) || mpsinputField0(mpsi) == NULL)
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }

   if (!strcmp(mpsinputField0(mpsi), "ROWS"))
      mpsinputSetSection(mpsi, MPS_ROWS);
   else if (!strcmp(mpsinputField0(mpsi), "OBJNAME"))
      mpsinputSetSection(mpsi, MPS_OBJNAME);
   else
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }

   return SCIP_OKAY;
}

/* Process OBJNAME section. This Section is an ILOG extension.
 */
static
SCIP_RETCODE readObjname(
   MPSINPUT*             mpsi
   )
{
   assert(mpsi != NULL);
   
   /* This has to be the Line with the name. */
   if (!mpsinputReadLine(mpsi) || mpsinputField1(mpsi) == NULL)
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }

   mpsinputSetObjname(mpsi, mpsinputField1(mpsi));
   
   /* Look for ROWS Section */
   if (!mpsinputReadLine(mpsi) || mpsinputField0(mpsi) == NULL)
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }
   if (strcmp(mpsinputField0(mpsi), "ROWS"))
   {
      mpsinputSyntaxerror(mpsi);
      return SCIP_OKAY;
   }

   mpsinputSetSection(mpsi, MPS_ROWS);

   return SCIP_OKAY;
}

/* Process ROWS section. 
 */
static 
SCIP_RETCODE readRows(
   MPSINPUT*             mpsi,
   SCIP*                 scip                /**< SCIP data structure */   
   )
{
   while(mpsinputReadLine(mpsi))
   {
      if (mpsinputField0(mpsi) != NULL)
      {
         /*printf("Objective name : %s\n", mpsinputObjname(mpsi));*/

         if (strcmp(mpsinputField0(mpsi), "COLUMNS"))
            break;

         mpsinputSetSection(mpsi, MPS_COLUMNS);

         return SCIP_OKAY;
      }
      if (*mpsinputField1(mpsi) == 'N')
      {
         if (*mpsinputObjname(mpsi) == '\0')
            mpsinputSetObjname(mpsi, mpsinputField2(mpsi));
      }
      else
      {
         SCIP_CONS* cons;
         SCIP_Bool dynamicrows;
         SCIP_Bool dynamicconss;

         cons = SCIPfindCons(scip, mpsinputField2(mpsi));
         if( cons != NULL )
            break;

         SCIP_CALL( SCIPgetBoolParam(scip, "reading/mpsreader/dynamicconss", &dynamicconss) );
         SCIP_CALL( SCIPgetBoolParam(scip, "reading/mpsreader/dynamicrows", &dynamicrows) );

         switch(*mpsinputField1(mpsi))
         {
         case 'G' :
            SCIP_CALL( SCIPcreateConsLinear(scip, &cons, mpsinputField2(mpsi), 0, NULL, NULL, 0.0, SCIPinfinity(scip), 
                  !dynamicrows, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, dynamicconss, dynamicrows) );
            break;
         case 'E' :
            SCIP_CALL( SCIPcreateConsLinear(scip, &cons, mpsinputField2(mpsi), 0, NULL, NULL, 0.0, 0.0, 
                  !dynamicrows, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, dynamicconss, dynamicrows) );
            break;
         case 'L' :
            SCIP_CALL( SCIPcreateConsLinear(scip, &cons, mpsinputField2(mpsi), 0, NULL, NULL, -SCIPinfinity(scip), 0.0,
                  !dynamicrows, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, dynamicconss, dynamicrows) );
            break;
         default :
            mpsinputSyntaxerror(mpsi);
            return SCIP_OKAY;
         }
         SCIP_CALL( SCIPaddCons(scip, cons) );
         SCIP_CALL( SCIPreleaseCons(scip, &cons) );
      }
   }
   mpsinputSyntaxerror(mpsi);

   return SCIP_OKAY;
}

/* Process COLUMNS section.
 */
static
SCIP_RETCODE readCols(
   MPSINPUT*             mpsi,
   SCIP*                 scip                /**< SCIP data structure */   
   ) 
{
   char          colname[MPS_MAX_LINELEN] = { '\0' };
   SCIP_CONS*    cons;
   SCIP_VAR*     var;
   SCIP_Real     val;

   var = NULL;
   while(mpsinputReadLine(mpsi))
   {
      if (mpsinputField0(mpsi) != 0)
      {
         if (strcmp(mpsinputField0(mpsi), "RHS"))
            break;

         /* add the last variable to the problem */
         if( var != NULL )
         {
            SCIP_CALL( SCIPaddVar(scip, var) );
            SCIP_CALL( SCIPreleaseVar(scip, &var) );
         }
         assert(var == NULL);

         mpsinputSetSection(mpsi, MPS_RHS);
         return SCIP_OKAY;
      }
      if (mpsinputField1(mpsi) == NULL || mpsinputField2(mpsi) == NULL || mpsinputField3(mpsi) == NULL)
         break;

      /* new column? */
      if (strcmp(colname, mpsinputField1(mpsi)))
      {
         SCIP_Bool dynamiccols;

         /* add the last variable to the problem */
         if( var != NULL )
         {
            SCIP_CALL( SCIPaddVar(scip, var) );
            SCIP_CALL( SCIPreleaseVar(scip, &var) );
         }
         assert(var == NULL);

         strcpy(colname, mpsinputField1(mpsi));

         SCIP_CALL( SCIPgetBoolParam(scip, "reading/mpsreader/dynamiccols", &dynamiccols) );

         if( mpsinputIsInteger(mpsi) )
         {
            /* for integer variables, default bounds are 0 <= x <= 1, and default cost is 0 */
            SCIP_CALL( SCIPcreateVar(scip, &var, colname, 0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY, !dynamiccols, dynamiccols,
                  NULL, NULL, NULL, NULL) );
         }
         else
         {
            /* for continuous variables, default bounds are 0 <= x, and default cost is 0 */
            SCIP_CALL( SCIPcreateVar(scip, &var, colname, 0.0, SCIPinfinity(scip), 0.0, SCIP_VARTYPE_CONTINUOUS,
                  !dynamiccols, dynamiccols, NULL, NULL, NULL, NULL) );
         }
      }
      assert(var != NULL);

      val = atof(mpsinputField3(mpsi));

      if (!strcmp(mpsinputField2(mpsi), mpsinputObjname(mpsi)))
      {
         SCIP_CALL( SCIPchgVarObj(scip, var, val) );
      }
      else 
      {
         cons = SCIPfindCons(scip, mpsinputField2(mpsi));
         if( cons == NULL )
            mpsinputEntryIgnored(scip, mpsi, "Column", mpsinputField1(mpsi), "row", mpsinputField2(mpsi));
         else if( !SCIPisZero(scip, val) )
         {
            SCIP_CALL( SCIPaddCoefLinear(scip, cons, var, val) );
         }
      }
      if (mpsinputField5(mpsi) != NULL)
      {
         assert(mpsinputField4(mpsi) != NULL);

         val = atof(mpsinputField5(mpsi));

         if (!strcmp(mpsinputField4(mpsi), mpsinputObjname(mpsi)))
         {
            SCIP_CALL( SCIPchgVarObj(scip, var, val) );
         }
         else 
         {
            cons = SCIPfindCons(scip, mpsinputField4(mpsi));
            if( cons == NULL )
               mpsinputEntryIgnored(scip, mpsi, "Column", mpsinputField1(mpsi), "row", mpsinputField4(mpsi));
            else if( !SCIPisZero(scip, val) )
            {
               SCIP_CALL( SCIPaddCoefLinear(scip, cons, var, val) );
            }
         }
      }
   }
   mpsinputSyntaxerror(mpsi);

   return SCIP_OKAY;
}

/* Process RHS section. 
 */
static
SCIP_RETCODE readRhs(
   MPSINPUT*             mpsi,
   SCIP*                 scip                /**< SCIP data structure */   
   )
{
   char        rhsname[MPS_MAX_LINELEN] = { '\0' };
   SCIP_CONS*  cons;
   SCIP_Real   lhs;
   SCIP_Real   rhs;
   SCIP_Real   val;

   while(mpsinputReadLine(mpsi))
   {
      if (mpsinputField0(mpsi) != NULL)
      {
         /*printf("RHS name       : %s\n", rhsname);*/

         if (!strcmp(mpsinputField0(mpsi), "RANGES"))
            mpsinputSetSection(mpsi, MPS_RANGES);
         else if (!strcmp(mpsinputField0(mpsi), "BOUNDS"))
            mpsinputSetSection(mpsi, MPS_BOUNDS);
         else if (!strcmp(mpsinputField0(mpsi), "ENDATA"))
            mpsinputSetSection(mpsi, MPS_ENDATA);
         else
            break;
         return SCIP_OKAY;
      }
      if ((mpsinputField2(mpsi) != NULL && mpsinputField3(mpsi) == NULL)
         || (mpsinputField4(mpsi) != NULL && mpsinputField5(mpsi) == NULL))
         mpsinputInsertName(mpsi, "_RHS_", FALSE);
      
      if (mpsinputField1(mpsi) == NULL || mpsinputField2(mpsi) == NULL || mpsinputField3(mpsi) == NULL)
         break;

      if (*rhsname == '\0')
         strcpy(rhsname, mpsinputField1(mpsi));
      
      if (!strcmp(rhsname, mpsinputField1(mpsi)))
      {
         cons = SCIPfindCons(scip, mpsinputField2(mpsi));
         if( cons == NULL )
            mpsinputEntryIgnored(scip, mpsi, "RHS", mpsinputField1(mpsi), "row", mpsinputField2(mpsi));
         else
         {
            val = atof(mpsinputField3(mpsi));

            /* find out the row sense */
            lhs = SCIPgetLhsLinear(scip, cons);
            rhs = SCIPgetRhsLinear(scip, cons);
            if( SCIPisInfinity(scip, -lhs) )
            {
               /* lhs = -infinity -> lower or equal */
               assert(SCIPisZero(scip, rhs));
               SCIP_CALL( SCIPchgRhsLinear(scip, cons, val) );
            }
            else if( SCIPisInfinity(scip, rhs) )
            {
               /* rhs = +infinity -> greater or equal */
               assert(SCIPisZero(scip, lhs));
               SCIP_CALL( SCIPchgLhsLinear(scip, cons, val) );
            }
            else
            {
               /* lhs > -infinity, rhs < infinity -> equality */
               assert(SCIPisZero(scip, lhs));
               assert(SCIPisZero(scip, rhs));
               SCIP_CALL( SCIPchgLhsLinear(scip, cons, val) );
               SCIP_CALL( SCIPchgRhsLinear(scip, cons, val) );
            }
         }
         if (mpsinputField5(mpsi) != NULL)
         {
            cons = SCIPfindCons(scip, mpsinputField4(mpsi));
            if( cons == NULL )
               mpsinputEntryIgnored(scip, mpsi, "RHS", mpsinputField1(mpsi), "row", mpsinputField4(mpsi));
            else
            {
               val = atof(mpsinputField5(mpsi));
               
               /* find out the row sense */
               lhs = SCIPgetLhsLinear(scip, cons);
               rhs = SCIPgetRhsLinear(scip, cons);
               if( SCIPisInfinity(scip, -lhs) )
               {
                  /* lhs = -infinity -> lower or equal */
                  assert(SCIPisZero(scip, rhs));
                  SCIP_CALL( SCIPchgRhsLinear(scip, cons, val) );
               }
               else if( SCIPisInfinity(scip, rhs) )
               {
                  /* rhs = +infinity -> greater or equal */
                  assert(SCIPisZero(scip, lhs));
                  SCIP_CALL( SCIPchgLhsLinear(scip, cons, val) );
               }
               else
               {
                  /* lhs > -infinity, rhs < infinity -> equality */
                  assert(SCIPisZero(scip, lhs));
                  assert(SCIPisZero(scip, rhs));
                  SCIP_CALL( SCIPchgLhsLinear(scip, cons, val) );
                  SCIP_CALL( SCIPchgRhsLinear(scip, cons, val) );
               }
            }
         }
      }
   }
   mpsinputSyntaxerror(mpsi);

   return SCIP_OKAY;
}

/*Process RANGES section
 */
static
SCIP_RETCODE readRanges(
   MPSINPUT*             mpsi,
   SCIP*                 scip                /**< SCIP data structure */   
   )
{
   char        rngname[MPS_MAX_LINELEN] = { '\0' };
   SCIP_CONS*  cons;
   SCIP_Real   lhs;
   SCIP_Real   rhs;
   SCIP_Real   val;

   while(mpsinputReadLine(mpsi))
   {
      if (mpsinputField0(mpsi) != NULL)
      {
         /*printf("Range name     : %s\n", rngname);*/

         if (!strcmp(mpsinputField0(mpsi), "BOUNDS"))
            mpsinputSetSection(mpsi, MPS_BOUNDS);
         else if (!strcmp(mpsinputField0(mpsi), "ENDATA"))
            mpsinputSetSection(mpsi, MPS_ENDATA);
         else
            break;
         return SCIP_OKAY;
      }
      if ((mpsinputField2(mpsi) != NULL && mpsinputField3(mpsi) == NULL)
         || (mpsinputField4(mpsi) != NULL && mpsinputField5(mpsi) == NULL))
         mpsinputInsertName(mpsi, "_RNG_", FALSE);

      if (mpsinputField1(mpsi) == NULL || mpsinputField2(mpsi) == NULL || mpsinputField3(mpsi) == NULL)
         break;

      if (*rngname == '\0')
         strcpy(rngname, mpsinputField1(mpsi));

      /* The rules are:
       * Row Sign   LHS             RHS
       * ----------------------------------------
       *  G   +/-   rhs             rhs + |range|
       *  L   +/-   rhs - |range|   rhs
       *  E   +     rhs             rhs + range
       *  E   -     rhs + range     rhs 
       * ----------------------------------------
       */  
      if (!strcmp(rngname, mpsinputField1(mpsi)))
      {
         cons = SCIPfindCons(scip, mpsinputField2(mpsi));
         if( cons == NULL )
            mpsinputEntryIgnored(scip, mpsi, "Range", mpsinputField1(mpsi), "row", mpsinputField2(mpsi));
         else
         {
            val = atof(mpsinputField3(mpsi));

            /* find out the row sense */
            lhs = SCIPgetLhsLinear(scip, cons);
            rhs = SCIPgetRhsLinear(scip, cons);
            if( SCIPisInfinity(scip, -lhs) )
            {
               /* lhs = -infinity -> lower or equal */
               SCIP_CALL( SCIPchgLhsLinear(scip, cons, rhs - REALABS(val)) );
            }
            else if( SCIPisInfinity(scip, rhs) )
            {
               /* rhs = +infinity -> greater or equal */
               SCIP_CALL( SCIPchgRhsLinear(scip, cons, lhs + REALABS(val)) );
            }
            else
            {
               /* lhs > -infinity, rhs < infinity -> equality */
               assert(SCIPisEQ(scip, lhs, rhs));
               if( val >= 0.0 )
               {
                  SCIP_CALL( SCIPchgRhsLinear(scip, cons, rhs + val) );
               }
               else
               {
                  SCIP_CALL( SCIPchgLhsLinear(scip, cons, lhs + val) );
               }
            }
         }
         if (mpsinputField5(mpsi) != NULL)
         {
            cons = SCIPfindCons(scip, mpsinputField4(mpsi));
            if( cons == NULL )
               mpsinputEntryIgnored(scip, mpsi, "Range", mpsinputField1(mpsi), "row", mpsinputField4(mpsi));
            else
            {
               val = atof(mpsinputField5(mpsi));

               /* find out the row sense */
               lhs = SCIPgetLhsLinear(scip, cons);
               rhs = SCIPgetRhsLinear(scip, cons);
               if( SCIPisInfinity(scip, -lhs) )
               {
                  /* lhs = -infinity -> lower or equal */
                  SCIP_CALL( SCIPchgLhsLinear(scip, cons, rhs - REALABS(val)) );
               }
               else if( SCIPisInfinity(scip, rhs) )
               {
                  /* rhs = +infinity -> greater or equal */
                  SCIP_CALL( SCIPchgRhsLinear(scip, cons, lhs + REALABS(val)) );
               }
               else
               {
                  /* lhs > -infinity, rhs < infinity -> equality */
                  assert(SCIPisEQ(scip, lhs, rhs));
                  if( val >= 0.0 )
                  {
                     SCIP_CALL( SCIPchgRhsLinear(scip, cons, rhs + val) );
                  }
                  else
                  {
                     SCIP_CALL( SCIPchgLhsLinear(scip, cons, lhs + val) );
                  }
               }
            }
         }
      }
   }
   mpsinputSyntaxerror(mpsi);

   return SCIP_OKAY;
}

/* Process BOUNDS section. 
 */
static
SCIP_RETCODE readBounds(
   MPSINPUT*             mpsi,
   SCIP*                 scip                /**< SCIP data structure */   
   )
{
   char        bndname[MPS_MAX_LINELEN] = { '\0' };
   SCIP_VAR*   var;
   SCIP_Real   val;

   while(mpsinputReadLine(mpsi))
   {
      if (mpsinputField0(mpsi) != 0)
      {
         /*printf("Bound name     : %s\n", bndname);*/

         if (strcmp(mpsinputField0(mpsi), "ENDATA"))
            break;

         mpsinputSetSection(mpsi, MPS_ENDATA);
         return SCIP_OKAY;
      }
      /* Is the value field used ? */
      if (  (!strcmp(mpsinputField1(mpsi), "LO"))
         || (!strcmp(mpsinputField1(mpsi), "UP"))
         || (!strcmp(mpsinputField1(mpsi), "FX"))
         || (!strcmp(mpsinputField1(mpsi), "LI"))
         || (!strcmp(mpsinputField1(mpsi), "UI")))
      {
         if (mpsinputField3(mpsi) != NULL && mpsinputField4(mpsi) == NULL)
            mpsinputInsertName(mpsi, "_BND_", TRUE);
      }
      else
      {
         if (mpsinputField2(mpsi) != NULL && mpsinputField3(mpsi) == NULL)
            mpsinputInsertName(mpsi, "_BND_", TRUE);
      }

      if (mpsinputField1(mpsi) == NULL || mpsinputField2(mpsi) == NULL || mpsinputField3(mpsi) == NULL)
         break;

      if (*bndname == '\0')
         strcpy(bndname, mpsinputField2(mpsi));
      
      /* Only read the first Bound in section */
      if (!strcmp(bndname, mpsinputField2(mpsi)))
      {
         var = SCIPfindVar(scip, mpsinputField3(mpsi));
         if( var == NULL )
            mpsinputEntryIgnored(scip, mpsi, "column", mpsinputField3(mpsi), "bound", bndname);
         else
         { 
            if( mpsinputField4(mpsi) == NULL )
               val = 0.0;
            else
               val = atof(mpsinputField4(mpsi));

            /* if a bound of a binary variable is given, the variable is converted into an integer variable
             * with default bounds 0 <= x <= infinity
             */
            if( SCIPvarGetType(var) == SCIP_VARTYPE_BINARY )
            {
               if( (mpsinputField1(mpsi)[1] == 'I') /* ILOG extension (Integer Bound) */
                  || (!(mpsinputField1(mpsi)[0] == 'L' && SCIPisEQ(scip, val, 0.0))
                     && !(mpsinputField1(mpsi)[0] == 'U' && SCIPisEQ(scip, val, 1.0))) )
               {
                  assert(SCIPisEQ(scip, SCIPvarGetLbGlobal(var), 0.0));
                  assert(SCIPisEQ(scip, SCIPvarGetUbGlobal(var), 1.0));
                  SCIP_CALL( SCIPchgVarType(scip, var, SCIP_VARTYPE_INTEGER) );
                  SCIP_CALL( SCIPchgVarUb(scip, var, SCIPinfinity(scip)) );
               }
            }

            switch(mpsinputField1(mpsi)[0])
            {
            case 'L':
               if( mpsinputField1(mpsi)[1] == 'I' ) /* ILOG extension (Integer Bound) */
               {
                  SCIP_CALL( SCIPchgVarType(scip, var, SCIP_VARTYPE_INTEGER) );
               }
               SCIP_CALL( SCIPchgVarLb(scip, var, val) );
               break;
            case 'U':
               if( mpsinputField1(mpsi)[1] == 'I' ) /* ILOG extension (Integer Bound) */
               {
                  SCIP_CALL( SCIPchgVarType(scip, var, SCIP_VARTYPE_INTEGER) );
               }
               SCIP_CALL( SCIPchgVarUb(scip, var, val) );
               break;
            case 'F':
               if (mpsinputField1(mpsi)[1] == 'X')
               {
                  SCIP_CALL( SCIPchgVarLb(scip, var, val) );
                  SCIP_CALL( SCIPchgVarUb(scip, var, val) );
               }
               else
               {
                  SCIP_CALL( SCIPchgVarLb(scip, var, -SCIPinfinity(scip)) );
                  SCIP_CALL( SCIPchgVarUb(scip, var, +SCIPinfinity(scip)) );
               }
               break;
            case 'M':
               SCIP_CALL( SCIPchgVarLb(scip, var, -SCIPinfinity(scip)) );
               break;
            case 'P':
               SCIP_CALL( SCIPchgVarUb(scip, var, +SCIPinfinity(scip)) );
               break;
            case 'B' : /* Ilog extension (Binary) */
               SCIP_CALL( SCIPchgVarLb(scip, var, 0.0) );
               SCIP_CALL( SCIPchgVarUb(scip, var, 1.0) );
               SCIP_CALL( SCIPchgVarType(scip, var, SCIP_VARTYPE_BINARY) );
               break;
            default:
               mpsinputSyntaxerror(mpsi);
               return SCIP_OKAY;
            }
         }
      }
   }
   mpsinputSyntaxerror(mpsi);

   return SCIP_OKAY;
}

/* Read LP in "MPS File Format".
 * 
 *  The specification is taken from the
 *
 *  IBM Optimization Library Guide and Reference
 *
 *  Online available at http://www.software.ibm.com/sos/features/libuser.htm
 *
 *  and from the 
 *
 *  ILOG CPLEX 7.0 Reference Manual, Appendix E, Page 531.
 *
 *  This routine should read all valid MPS format files. 
 *  What it will not do, is find all cases where a file is ill formed. 
 *  If this happens it may complain and read nothing or read "something".
 */  
static
SCIP_RETCODE readMps(
   SCIP*                 scip,               /**< SCIP data structure */   
   const char*           filename            /**< name of the input file */
   )
{
   SCIP_FILE* fp;
   MPSINPUT* mpsi;
   SCIP_Bool error;

   assert(scip != NULL);
   assert(filename != NULL);

   if (NULL == (fp = SCIPfopen(filename, "r")))
   {
      SCIPerrorMessage("cannot open file <%s> for reading\n", filename);
      perror(filename);
      return SCIP_NOFILE;
   }   

   SCIP_CALL( mpsinputCreate(scip, &mpsi, fp) );

   SCIP_CALL( readName(mpsi) );

   SCIP_CALL( SCIPcreateProb(scip, mpsi->probname, NULL, NULL, NULL, NULL, NULL, NULL) );

   if (mpsinputSection(mpsi) == MPS_OBJSEN)
   {
      SCIP_CALL( readObjsen(mpsi) );
   }
   if (mpsinputSection(mpsi) == MPS_OBJNAME)
   {
      SCIP_CALL( readObjname(mpsi) );
   }
   if (mpsinputSection(mpsi) == MPS_ROWS)
   {
      SCIP_CALL( readRows(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) == MPS_COLUMNS)
   {
      SCIP_CALL( readCols(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) == MPS_RHS)
   {
      SCIP_CALL( readRhs(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) == MPS_RANGES)
   {
      SCIP_CALL( readRanges(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) == MPS_BOUNDS)
   {
      SCIP_CALL( readBounds(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) != MPS_ENDATA)
      mpsinputSyntaxerror(mpsi);

   SCIPfclose(fp);

   error = mpsinputHasError(mpsi);

   if( !error )
   {
      SCIP_CALL( SCIPsetObjsense(scip, mpsinputObjsense(mpsi)) );

      /*printf("Objective sense: %s\n", (mpsinputObjsense(mpsi) == SCIP_OBJSENSE_MINIMIZE) ? "Minimize" : "Maximize");*/
   }
   mpsinputFree(scip, &mpsi);

   if( error )
      return SCIP_PARSEERROR;
   else
      return SCIP_OKAY;
}




/*
 * Callback methods of reader
 */

/** destructor of reader to free user data (called when SCIP is exiting) */
#define readerFreeMps NULL


/** problem reading method of reader */
static
SCIP_DECL_READERREAD(readerReadMps)
{  /*lint --e{715}*/
   assert(reader != NULL);
   assert(strcmp(SCIPreaderGetName(reader), READER_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   SCIP_CALL( readMps(scip, filename) );

   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}




/*
 * mps file reader specific interface methods
 */

/** includes the mps file reader in SCIP */
SCIP_RETCODE SCIPincludeReaderMps(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_READERDATA* readerdata;

   /* create mps reader data */
   readerdata = NULL;

   /* include mps reader */
   SCIP_CALL( SCIPincludeReader(scip, READER_NAME, READER_DESC, READER_EXTENSION,
         readerFreeMps, readerReadMps, readerdata) );

   /* add mps reader parameters */
   SCIP_CALL( SCIPaddBoolParam(scip,
         "reading/mpsreader/dynamicconss", "should model constraints be subject to aging?",
         NULL, TRUE, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "reading/mpsreader/dynamiccols", "should columns be added and removed dynamically to the LP?",
         NULL, FALSE, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "reading/mpsreader/dynamicrows", "should rows be added and removed dynamically to the LP?",
         NULL, FALSE, NULL, NULL) );
   
   return SCIP_OKAY;
}

