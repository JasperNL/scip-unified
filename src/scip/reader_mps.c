/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2002 Tobias Achterberg                              */
/*                            Thorsten Koch                                  */
/*                            Alexander Martin                               */
/*                  2002-2002 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the SCIP Academic Licence.        */
/*                                                                           */
/*  You should have received a copy of the SCIP Academic License             */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   reader_mps.c
 * @brief  MPS file reader
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "reader_mps.h"
#include "cons_linear.h"


#define READER_NAME             "MPSreader"
#define READER_DESC             "MPS file reader"
#define READER_EXTENSION        "mps"



/*
 * MPS reader internal methods
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
   MPSSECTION      section;
   FILE*           fp;
   int             lineno;
   OBJSENSE        objsense;
   Bool            haserror;
   char            buf[MPS_MAX_LINELEN];
   const char*     f0;
   const char*     f1;
   const char*     f2;
   const char*     f3;
   const char*     f4;
   const char*     f5;
   char            probname[MPS_MAX_LINELEN];
   char            objname [MPS_MAX_LINELEN];
   Bool            isinteger;
};
typedef struct MpsInput MPSINPUT;



static
RETCODE mpsinputCreate(
   MPSINPUT**       mpsi,
   FILE*            fp
   )
{
   assert(mpsi != NULL);
   assert(fp != NULL);

   ALLOC_OKAY( allocMemory(*mpsi) );

   (*mpsi)->section     = MPS_NAME;
   (*mpsi)->fp          = fp;
   (*mpsi)->lineno      = 0;
   (*mpsi)->objsense    = SCIP_OBJSENSE_MINIMIZE;
   (*mpsi)->haserror    = FALSE;
   (*mpsi)->isinteger   = FALSE;
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
   MPSINPUT**       mpsi
   )
{
   freeMemory(*mpsi);
}

static
MPSSECTION mpsinputSection(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->section;
}

static
int mpsinputLineno(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->lineno;
}

static
const char* mpsinputField0(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f0;
}

static
const char* mpsinputField1(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f1;
}

static
const char* mpsinputField2(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f2;
}

static
const char* mpsinputField3(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f3;
}

static
const char* mpsinputField4(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f4;
}

static
const char* mpsinputField5(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->f5;
}

static
const char* mpsinputProbname(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->probname;
}

static
const char* mpsinputObjname(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->objname;
}

static
OBJSENSE mpsinputObjsense(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->objsense;
}

static
Bool mpsinputHasError(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->haserror;
}

static
Bool mpsinputIsInteger(
   const MPSINPUT*  mpsi
   )
{
   assert(mpsi != NULL);

   return mpsi->isinteger;
}

static
void mpsinputSetSection(
   MPSINPUT*        mpsi,
   MPSSECTION       section
   )
{
   assert(mpsi != NULL);

   mpsi->section = section;
}

static
void mpsinputSetProbname(
   MPSINPUT*        mpsi,
   const char*      probname
   )
{
   assert(mpsi     != NULL);
   assert(probname != NULL);
   assert(strlen(probname) < sizeof(mpsi->probname));
   
   strcpy(mpsi->probname, probname);
}

static
void mpsinputSetObjname(
   MPSINPUT*        mpsi, 
   const char*      objname
   )
{
   assert(mpsi != NULL);
   assert(objname != NULL);
   assert(strlen(objname) < sizeof(mpsi->objname));

   strcpy(mpsi->objname, objname);
}

static
void mpsinputSetObjsense(
   MPSINPUT*        mpsi,
   OBJSENSE         sense
   )
{
   assert(mpsi != NULL);

   mpsi->objsense = sense;
}

static
void mpsinputSyntaxerror(
   MPSINPUT*        mpsi
   ) 
{
   assert(mpsi != NULL);

   fprintf(stderr, "Syntax error in line %d\n", mpsi->lineno);
   mpsi->section  = MPS_ENDATA;
   mpsi->haserror = TRUE;
}

static
void mpsinputEntryIgnored(
   MPSINPUT*        mpsi, 
   const char*      what, 
   const char*      what_name, 
   const char*      entity, 
   const char*      entity_name
   )
{
   assert(mpsi        != NULL);
   assert(what        != NULL);
   assert(what_name   != NULL);
   assert(entity      != NULL);
   assert(entity_name != NULL);

   fprintf(stderr, "Warning line %d : %s \"%s\" for %s \"%s\" ignored\n",
      mpsi->lineno, what, what_name, entity, entity_name);
}

/* fill the line from \p pos up to column 80 with blanks.
 */
static
void clearFrom(
   char*            buf,
   int              pos
   )
{
   int i;

   for(i = pos; i < 80; i++)
      buf[i] = BLANK;
   buf[80] = '\0';
}

/* change all blanks inside a field to #PATCH_CHAR.
 */
static
void patchField(
   char*            buf,
   int              beg,
   int              end
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

/* read a MPS format data line and parse the fields.
 */
static
Bool mpsinputReadLine(
   MPSINPUT*        mpsi
   )
{
   unsigned int len;
   unsigned int i;
   int   space;
   char* s;
   Bool  is_marker;

   do
   {
      mpsi->f0  = mpsi->f1 = mpsi->f2 = mpsi->f3 = mpsi->f4 = mpsi->f5 = 0;
      is_marker = FALSE;
   
      // Read until we have a not comment line.
      do
      {
         if (NULL == fgets(mpsi->buf, sizeof(mpsi->buf), mpsi->fp))
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
         mpsi->f0 = strtok(&mpsi->buf[0], " ");

         assert(mpsi->f0 != 0);

         mpsi->f1 = strtok(NULL, " ");

         return TRUE;
      }

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
         /* We assume fixed format, so we patch possible embedded spaces.
          */
         patchField(mpsi->buf,  4, 12);
         patchField(mpsi->buf, 14, 22);
         patchField(mpsi->buf, 39, 47);
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
         if (NULL == (mpsi->f1 = strtok(s, " ")))
            break;
         
         if ((NULL == (mpsi->f2 = strtok(NULL, " "))) || (*mpsi->f2 == '$'))
         {
            mpsi->f2 = 0;
            break;      
         }
         if (!strcmp(mpsi->f2, "'MARKER'"))
            is_marker = TRUE;
            
         if ((NULL == (mpsi->f3 = strtok(NULL, " "))) || (*mpsi->f3 == '$'))
         {
            mpsi->f3 = 0;
            break;      
         }
         if( is_marker )
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

         if ((NULL == (mpsi->f4 = strtok(NULL, " "))) || (*mpsi->f4 == '$'))
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

         if ((NULL == (mpsi->f5 = strtok(NULL, " "))) || (*mpsi->f5 == '$'))
            mpsi->f5 = 0;
      }
      while(FALSE);
   }
   while(is_marker);

#if 0
   debugMessage("-----------------------------------------------\n");
   debugMessage("f0=%s\n", (mpsi->f0 == NULL ? "nil" : mpsi->f0));
   debugMessage("f1=%s\n", (mpsi->f1 == NULL ? "nil" : mpsi->f1));
   debugMessage("f2=%s\n", (mpsi->f2 == NULL ? "nil" : mpsi->f2));
   debugMessage("f3=%s\n", (mpsi->f3 == NULL ? "nil" : mpsi->f3));
   debugMessage("f4=%s\n", (mpsi->f4 == NULL ? "nil" : mpsi->f4));
   debugMessage("f5=%s\n", (mpsi->f5 == NULL ? "nil" : mpsi->f5));
   debugMessage("-----------------------------------------------\n");
#endif

   return TRUE;
}

/* Insert \p name as field 1 and shift all other fields up.
 */
static
void mpsinputInsertName(
   MPSINPUT*        mpsi,
   const char*      name,
   Bool             second
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
RETCODE readName(
   MPSINPUT*        mpsi
   )
{
   assert(mpsi != NULL);

   do
   {
      /* This has to be the Line with the NAME section.
       */
      if (!mpsinputReadLine(mpsi) || mpsinputField0(mpsi) == NULL || strcmp(mpsinputField0(mpsi), "NAME"))
         break;

      // Sometimes the name is omitted.
      mpsinputSetProbname(mpsi, (mpsinputField1(mpsi) == 0) ? "_MPS_" : mpsinputField1(mpsi));

      printf("Problem name   : %s\n", mpsinputProbname(mpsi));
 
      // This hat to be a new section
      if (!mpsinputReadLine(mpsi) || (mpsinputField0(mpsi) == NULL))
         break;

      if (!strcmp(mpsinputField0(mpsi), "ROWS"))
         mpsinputSetSection(mpsi, MPS_ROWS);
      else if (!strcmp(mpsinputField0(mpsi), "OBJSEN"))
         mpsinputSetSection(mpsi, MPS_OBJSEN);
      else if (!strcmp(mpsinputField0(mpsi), "OBJNAME"))
         mpsinputSetSection(mpsi, MPS_OBJNAME);
      else
         break;

      return SCIP_OKAY;
   }
   while(FALSE);

   mpsinputSyntaxerror(mpsi);

   return SCIP_OKAY;
}

/* Process OBJSEN section. This Section is an ILOG extension.
 */
static
RETCODE readObjsen(
   MPSINPUT*        mpsi
   )
{
   assert(mpsi != NULL);

   do
   {
      /* This has to be the Line with MIN or MAX. */
      if (!mpsinputReadLine(mpsi) || (mpsinputField1(mpsi) == NULL))
         break;

      if (strcmp(mpsinputField1(mpsi), "MIN"))
         mpsinputSetObjsense(mpsi, SCIP_OBJSENSE_MINIMIZE);
      else if (strcmp(mpsinputField1(mpsi), "MAX"))
         mpsinputSetObjsense(mpsi, SCIP_OBJSENSE_MAXIMIZE);
      else
         break;

      /* Look for ROWS or OBJNAME Section */
      if (!mpsinputReadLine(mpsi) || mpsinputField0(mpsi) == NULL)
         break;

      if (!strcmp(mpsinputField0(mpsi), "ROWS"))
         mpsinputSetSection(mpsi, MPS_ROWS);
      else if (!strcmp(mpsinputField0(mpsi), "OBJNAME"))
         mpsinputSetSection(mpsi, MPS_OBJNAME);
      else
         break;

      return SCIP_OKAY;
   }
   while(FALSE);

   mpsinputSyntaxerror(mpsi);

   return SCIP_OKAY;
}

/* Process OBJNAME section. This Section is an ILOG extension.
 */
static
RETCODE readObjname(
   MPSINPUT*        mpsi
   )
{
   assert(mpsi != NULL);

   do
   {
      // This has to be the Line with the name.
      if (!mpsinputReadLine(mpsi) || mpsinputField1(mpsi) == NULL)
         break;

      mpsinputSetObjname(mpsi, mpsinputField1(mpsi));

      // Look for ROWS Section
      if (!mpsinputReadLine(mpsi) || mpsinputField0(mpsi) == NULL)
         break;

      if (strcmp(mpsinputField0(mpsi), "ROWS"))
         break;

      mpsinputSetSection(mpsi, MPS_ROWS);
      return SCIP_OKAY;
   }
   while(FALSE);

   mpsinputSyntaxerror(mpsi);

   return SCIP_OKAY;
}

/* Process ROWS section. 
 */
static 
RETCODE readRows(
   MPSINPUT*        mpsi,
   SCIP*            scip                /**< SCIP data structure */   
   )
{
   while(mpsinputReadLine(mpsi))
   {
      if (mpsinputField0(mpsi) != NULL)
      {
         printf("Objective name : %s\n", mpsinputObjname(mpsi));

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
         CONS* cons;
   
         CHECK_OKAY( SCIPfindCons(scip, mpsinputField2(mpsi), &cons) );
         if( cons != NULL )
            break;

         switch(*mpsinputField1(mpsi))
         {
         case 'G' :
            CHECK_SCIP( SCIPcreateConsLinear(scip, &cons, mpsinputField2(mpsi), 0, NULL, NULL, 
                           0.0, SCIPinfinity(scip), TRUE, FALSE) );
            break;
         case 'E' :
            CHECK_SCIP( SCIPcreateConsLinear(scip, &cons, mpsinputField2(mpsi), 0, NULL, NULL, 
                           0.0, 0.0, TRUE, FALSE) );
            break;
         case 'L' :
            CHECK_SCIP( SCIPcreateConsLinear(scip, &cons, mpsinputField2(mpsi), 0, NULL, NULL,
                           -SCIPinfinity(scip), 0.0, TRUE, FALSE) );
            break;
         default :
            mpsinputSyntaxerror(mpsi);
            return SCIP_OKAY;
         }
         CHECK_SCIP( SCIPaddCons(scip, cons) );
         CHECK_SCIP( SCIPreleaseCons(scip, &cons) );
      }
   }
   mpsinputSyntaxerror(mpsi);

   return SCIP_OKAY;
}

/* Process COLUMNS section.
 */
static
RETCODE readCols(
   MPSINPUT*        mpsi,
   SCIP*            scip                /**< SCIP data structure */   
   ) 
{
   char     colname[MPS_MAX_LINELEN] = { '\0' };
   CONS*    cons;
   VAR*     var;
   Real     val;

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
            CHECK_SCIP( SCIPaddVar(scip, var) );
            CHECK_SCIP( SCIPreleaseVar(scip, &var) );
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
         /* add the last variable to the problem */
         if( var != NULL )
         {
            CHECK_SCIP( SCIPaddVar(scip, var) );
            CHECK_SCIP( SCIPreleaseVar(scip, &var) );
         }
         assert(var == NULL);

         strcpy(colname, mpsinputField1(mpsi));

         if( mpsinputIsInteger(mpsi) )
         {
            /* for integer variables, default bounds are 0 <= x <= 1, and default cost is 0 */
            CHECK_SCIP( SCIPcreateVar(scip, &var, colname, 0.0, 1.0, 0.0, SCIP_VARTYPE_INTEGER) );
         }
         else
         {
            /* for continous variables, default bounds are 0 <= x, and default cost is 0 */
            CHECK_SCIP( SCIPcreateVar(scip, &var, colname, 0.0, SCIPinfinity(scip), 0.0, SCIP_VARTYPE_CONTINOUS) );
         }
      }
      assert(var != NULL);

      val = atof(mpsinputField3(mpsi));

      if (!strcmp(mpsinputField2(mpsi), mpsinputObjname(mpsi)))
      {
         CHECK_OKAY( SCIPvarChgObj(var, val) );
      }
      else 
      {
         CHECK_OKAY( SCIPfindCons(scip, mpsinputField2(mpsi), &cons) );
         if( cons == NULL )
            mpsinputEntryIgnored(mpsi, "Column", mpsinputField1(mpsi), "row", mpsinputField2(mpsi));
         else if( !SCIPisZero(scip, val) )
         {
            CHECK_OKAY( SCIPconsLinearAddCoef(cons, scip, var, val) );
         }
      }
      if (mpsinputField5(mpsi) != NULL)
      {
         assert(mpsinputField4(mpsi) != NULL);

         val = atof(mpsinputField5(mpsi));

         if (!strcmp(mpsinputField4(mpsi), mpsinputObjname(mpsi)))
         {
            CHECK_OKAY( SCIPvarChgObj(var, val) );
         }
         else 
         {
            CHECK_OKAY( SCIPfindCons(scip, mpsinputField4(mpsi), &cons) );
            if( cons == NULL )
               mpsinputEntryIgnored(mpsi, "Column", mpsinputField1(mpsi), "row", mpsinputField4(mpsi));
            else if( !SCIPisZero(scip, val) )
            {
               CHECK_OKAY( SCIPconsLinearAddCoef(cons, scip, var, val) );
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
RETCODE readRhs(
   MPSINPUT*        mpsi,
   SCIP*            scip                /**< SCIP data structure */   
   )
{
   char   rhsname[MPS_MAX_LINELEN] = { '\0' };
   CONS*  cons;
   Real   lhs;
   Real   rhs;
   Real   val;

   while(mpsinputReadLine(mpsi))
   {
      if (mpsinputField0(mpsi) != NULL)
      {
         printf("RHS name       : %s\n", rhsname);

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
         CHECK_OKAY( SCIPfindCons(scip, mpsinputField2(mpsi), &cons) );
         if( cons == NULL )
            mpsinputEntryIgnored(mpsi, "RHS", mpsinputField1(mpsi), "row", mpsinputField2(mpsi));
         else
         {
            val = atof(mpsinputField3(mpsi));

            /* find out the row sense */
            CHECK_OKAY( SCIPconsLinearGetLhs(cons, scip, &lhs) );
            CHECK_OKAY( SCIPconsLinearGetRhs(cons, scip, &rhs) );
            if( SCIPisInfinity(scip, -lhs) )
            {
               /* lhs = -infinity -> lower or equal */
               assert(SCIPisZero(scip, rhs));
               CHECK_OKAY( SCIPconsLinearChgRhs(cons, scip, val) );
            }
            else if( SCIPisInfinity(scip, rhs) )
            {
               /* rhs = +infinity -> greater or equal */
               assert(SCIPisZero(scip, lhs));
               CHECK_OKAY( SCIPconsLinearChgLhs(cons, scip, val) );
            }
            else
            {
               /* lhs > -infinity, rhs < infinity -> equality */
               assert(SCIPisZero(scip, lhs));
               assert(SCIPisZero(scip, rhs));
               CHECK_OKAY( SCIPconsLinearChgLhs(cons, scip, val) );
               CHECK_OKAY( SCIPconsLinearChgRhs(cons, scip, val) );
            }
         }
      }
      if (mpsinputField5(mpsi) != NULL)
      {
         CHECK_OKAY( SCIPfindCons(scip, mpsinputField4(mpsi), &cons) );
         if( cons == NULL )
            mpsinputEntryIgnored(mpsi, "RHS", mpsinputField1(mpsi), "row", mpsinputField4(mpsi));
         else
         {
            val = atof(mpsinputField5(mpsi));
         
            /* find out the row sense */
            CHECK_OKAY( SCIPconsLinearGetLhs(cons, scip, &lhs) );
            CHECK_OKAY( SCIPconsLinearGetRhs(cons, scip, &rhs) );
            if( SCIPisInfinity(scip, -lhs) )
            {
               /* lhs = -infinity -> lower or equal */
               assert(SCIPisZero(scip, rhs));
               CHECK_OKAY( SCIPconsLinearChgRhs(cons, scip, val) );
            }
            else if( SCIPisInfinity(scip, rhs) )
            {
               /* rhs = +infinity -> greater or equal */
               assert(SCIPisZero(scip, lhs));
               CHECK_OKAY( SCIPconsLinearChgLhs(cons, scip, val) );
            }
            else
            {
               /* lhs > -infinity, rhs < infinity -> equality */
               assert(SCIPisZero(scip, lhs));
               assert(SCIPisZero(scip, rhs));
               CHECK_OKAY( SCIPconsLinearChgLhs(cons, scip, val) );
               CHECK_OKAY( SCIPconsLinearChgRhs(cons, scip, val) );
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
RETCODE readRanges(
   MPSINPUT*        mpsi,
   SCIP*            scip                /**< SCIP data structure */   
   )
{
   char   rngname[MPS_MAX_LINELEN] = { '\0' };
   CONS*  cons;
   Real   lhs;
   Real   rhs;
   Real   val;

   while(mpsinputReadLine(mpsi))
   {
      if (mpsinputField0(mpsi) != NULL)
      {
         printf("Range name     : %s\n", rngname);

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
         CHECK_OKAY( SCIPfindCons(scip, mpsinputField2(mpsi), &cons) );
         if( cons == NULL )
            mpsinputEntryIgnored(mpsi, "Range", mpsinputField1(mpsi), "row", mpsinputField2(mpsi));
         else
         {
            val = atof(mpsinputField3(mpsi));

            /* find out the row sense */
            CHECK_OKAY( SCIPconsLinearGetLhs(cons, scip, &lhs) );
            CHECK_OKAY( SCIPconsLinearGetRhs(cons, scip, &rhs) );
            if( SCIPisInfinity(scip, -lhs) )
            {
               /* lhs = -infinity -> lower or equal */
               CHECK_OKAY( SCIPconsLinearChgLhs(cons, scip, rhs - ABS(val)) );
            }
            else if( SCIPisInfinity(scip, rhs) )
            {
               /* rhs = +infinity -> greater or equal */
               CHECK_OKAY( SCIPconsLinearChgRhs(cons, scip, lhs + ABS(val)) );
            }
            else
            {
               /* lhs > -infinity, rhs < infinity -> equality */
               assert(SCIPisEQ(scip, lhs, rhs));
               if( val >= 0.0 )
               {
                  CHECK_OKAY( SCIPconsLinearChgRhs(cons, scip, rhs + val) );
               }
               else
               {
                  CHECK_OKAY( SCIPconsLinearChgLhs(cons, scip, lhs + val) );
               }
            }
         }
         if (mpsinputField5(mpsi) != NULL)
         {
            CHECK_OKAY( SCIPfindCons(scip, mpsinputField4(mpsi), &cons) );
            if( cons == NULL )
               mpsinputEntryIgnored(mpsi, "Range", mpsinputField1(mpsi), "row", mpsinputField4(mpsi));
            else
            {
               val = atof(mpsinputField5(mpsi));

               /* find out the row sense */
               CHECK_OKAY( SCIPconsLinearGetLhs(cons, scip, &lhs) );
               CHECK_OKAY( SCIPconsLinearGetRhs(cons, scip, &rhs) );
               if( SCIPisInfinity(scip, -lhs) )
               {
                  /* lhs = -infinity -> lower or equal */
                  CHECK_OKAY( SCIPconsLinearChgLhs(cons, scip, rhs - ABS(val)) );
               }
               else if( SCIPisInfinity(scip, rhs) )
               {
                  /* rhs = +infinity -> greater or equal */
                  CHECK_OKAY( SCIPconsLinearChgRhs(cons, scip, lhs + ABS(val)) );
               }
               else
               {
                  /* lhs > -infinity, rhs < infinity -> equality */
                  assert(SCIPisEQ(scip, lhs, rhs));
                  if( val >= 0.0 )
                  {
                     CHECK_OKAY( SCIPconsLinearChgRhs(cons, scip, rhs + val) );
                  }
                  else
                  {
                     CHECK_OKAY( SCIPconsLinearChgLhs(cons, scip, lhs + val) );
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
RETCODE readBounds(
   MPSINPUT*        mpsi,
   SCIP*            scip                /**< SCIP data structure */   
   )
{
   char   bndname[MPS_MAX_LINELEN] = { '\0' };
   VAR*   var;
   Real   val;

   while(mpsinputReadLine(mpsi))
   {
      if (mpsinputField0(mpsi) != 0)
      {
         printf("Bound name     : %s\n", bndname);

         if (strcmp(mpsinputField0(mpsi), "ENDATA"))
            break;

         mpsinputSetSection(mpsi, MPS_ENDATA);
         return SCIP_OKAY;
      }
      // Is the value field used ?
      if (  (!strcmp(mpsinputField1(mpsi), "LO"))
         || (!strcmp(mpsinputField1(mpsi), "UP"))
         || (!strcmp(mpsinputField1(mpsi), "FX"))
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
         CHECK_OKAY( SCIPfindVar(scip, mpsinputField3(mpsi), &var) );
         if( var == NULL )
            mpsinputEntryIgnored(mpsi, "column", mpsinputField3(mpsi), "bound", bndname);
         else
         { 
            if( mpsinputField4(mpsi) == NULL )
               val = 0.0;
            else
               val = atof(mpsinputField4(mpsi));

            switch(*mpsinputField1(mpsi))
            {
            case 'L':
               /* ILOG extension (Integer Lower Bound)
                */
               if (mpsinputField1(mpsi)[1] == 'I')
               {
                  CHECK_OKAY( SCIPvarChgType(var, SCIP_VARTYPE_INTEGER) );
               }

               CHECK_OKAY( SCIPchgLb(scip, var, val) );
               break;
            case 'U':
               /* ILOG extension (Integer Upper Bound)
                */
               if (mpsinputField1(mpsi)[1] == 'I')
               {
                  CHECK_OKAY( SCIPvarChgType(var, SCIP_VARTYPE_INTEGER) );
               }

               CHECK_OKAY( SCIPchgUb(scip, var, val) );               
               break;
            case 'F':
               if (mpsinputField1(mpsi)[1] == 'X')
               {
                  CHECK_OKAY( SCIPchgLb(scip, var, val) );
                  CHECK_OKAY( SCIPchgUb(scip, var, val) );
               }
               else
               {
                  CHECK_OKAY( SCIPchgLb(scip, var, -SCIPinfinity(scip)) );
                  CHECK_OKAY( SCIPchgUb(scip, var, +SCIPinfinity(scip)) );
               }
               break;
            case 'M':
               CHECK_OKAY( SCIPchgLb(scip, var, -SCIPinfinity(scip)) );
               break;
            case 'P':
               CHECK_OKAY( SCIPchgUb(scip, var, +SCIPinfinity(scip)) );
               break;
            case 'B' : // Ilog extension (Binary)
               CHECK_OKAY( SCIPchgLb(scip, var, 0.0) );
               CHECK_OKAY( SCIPchgUb(scip, var, 1.0) );
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
 *
 *  @return TRUE if the file was read correctly.
 */  
static
RETCODE readMPS(
   SCIP*            scip,               /**< SCIP data structure */   
   const char*      filename
   )
{
   FILE*     fp;
   MPSINPUT* mpsi;
   Bool      error;

   assert(scip != NULL);
   assert(filename != NULL);

   if (NULL == (fp = fopen(filename, "r")))
   {
      char s[1024];
      sprintf(s, "cannot open file <%s>", filename);
      errorMessage(s);
      perror(filename);
      return SCIP_READERR;
   }   

   CHECK_OKAY( mpsinputCreate(&mpsi, fp) );

   CHECK_OKAY( readName(mpsi) );

   if (mpsinputSection(mpsi) == MPS_OBJSEN)
   {
      CHECK_OKAY( readObjsen(mpsi) );
   }
   if (mpsinputSection(mpsi) == MPS_OBJNAME)
   {
      CHECK_OKAY( readObjname(mpsi) );
   }
   if (mpsinputSection(mpsi) == MPS_ROWS)
   {
      CHECK_OKAY( readRows(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) == MPS_COLUMNS)
   {
      CHECK_OKAY( readCols(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) == MPS_RHS)
   {
      CHECK_OKAY( readRhs(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) == MPS_RANGES)
   {
      CHECK_OKAY( readRanges(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) == MPS_BOUNDS)
   {
      CHECK_OKAY( readBounds(mpsi, scip) );
   }
   if (mpsinputSection(mpsi) != MPS_ENDATA)
      mpsinputSyntaxerror(mpsi);

   fclose(fp);

   error = mpsinputHasError(mpsi);

   if( !error )
   {
      CHECK_OKAY( SCIPsetObjsense(scip, mpsinputObjsense(mpsi)) );

      printf("Objective sense: %s\n", (mpsinputObjsense(mpsi) == SCIP_OBJSENSE_MINIMIZE) ? "Minimize" : "Maximize"); 
   }
   mpsinputFree(&mpsi);

   if( error )
      return SCIP_READERR;
   else
      return SCIP_OKAY;
}



/*
 * Callback methods
 */

static
DECL_READERREAD(SCIPreaderReadMPS)
{
   assert(reader != NULL);
   assert(strcmp(SCIPreaderGetName(reader), READER_NAME) == 0);
   assert(scip != NULL);

   CHECK_OKAY( readMPS(scip, filename) );

   return SCIP_OKAY;
}




/*
 * MPS file reader specific interface methods
 */

RETCODE SCIPincludeReaderMPS(           /**< includes the MPS file reader in SCIP */
   SCIP*            scip                /**< SCIP data structure */
   )
{
   CHECK_OKAY( SCIPincludeReader(scip, READER_NAME, READER_DESC, READER_EXTENSION,
                  NULL, NULL, SCIPreaderReadMPS, NULL) );

   return SCIP_OKAY;
}

