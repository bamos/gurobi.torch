#include <gurobi_c.h>

#include <TH/TH.h>

#undef NDEBUG
#include <assert.h>
#include <string.h>

void* /* GRBenv* */ GT_loadenv(const char* logfilename, int outputFlag) {
  GRBenv *env;
  int error = GRBloadenv(&env, logfilename);
  assert(!error);

  int OutputFlag = 0;
  error = GRBsetintparam(env, "OutputFlag", outputFlag);
  assert(!error);

  return env;
}

void* /* GRBmodel* */ GT_newmodel(void *env, const char *name, THDoubleTensor *obj,
                               THDoubleTensor *lb, THDoubleTensor *ub) {
  GRBmodel *model;
  int nVars = THDoubleTensor_size(obj, 0);
  double *obj_ = THDoubleTensor_data(obj);
  double *lb_ = THDoubleTensor_data(lb);

  double *ub_ = 0;
  if (ub) {
    ub_ = THDoubleTensor_data(ub);
  }

  int error = GRBnewmodel(env, &model, name, nVars, obj_, lb_, ub_, 0, 0);
  assert(!error);

  return model;
}

int GT_setdblattrlist(void *model, const char *name, int len, THIntTensor *ind,
                      THDoubleTensor *values) {
  int *ind_ = THIntTensor_data(ind);
  double *values_ = THDoubleTensor_data(values);
  int error = GRBsetdblattrlist(model, name, len, ind_, values_);
  assert(!error);
}

int GT_getintattr(void *model, const char *name) {
  GRBmodel *model_ = (GRBmodel*) model;
  int attr;
  int error = GRBgetintattr(model, name, &attr);
  assert(!error);
  return attr;
}

void GT_addconstr(void *model, int nnz, THIntTensor *cind, THDoubleTensor *cval,
                  const char *sense, double rhs) {
  GRBmodel *model_ = (GRBmodel*) model;
  int* cind_ = THIntTensor_data(cind);
  double* cval_ = THDoubleTensor_data(cval);

  char sense_;
  if (!strcmp(sense, "LE")) {
    sense_ = GRB_LESS_EQUAL;
  } else if (!strcmp(sense, "EQ")) {
    sense_ = GRB_EQUAL;
  } else if (!strcmp(sense, "GE")) {
    sense_ = GRB_GREATER_EQUAL;
  } else {
    printf("WARNING: sense incorrectly set\n");
    assert(0);
  }

  int error = GRBaddconstr(model_, nnz, cind_, cval_, sense_, rhs, 0);
  assert(!error);
}

void GT_addqpterms(void *model, int numqnz, THIntTensor *qrow, THIntTensor *qcol,
                   THDoubleTensor *qval) {
  int *qrow_ = THIntTensor_data(qrow);
  int *qcol_ = THIntTensor_data(qcol);
  double *qval_ = THDoubleTensor_data(qval);
  int error = GRBaddqpterms(model, numqnz, qrow_, qcol_, qval_);
  assert(!error);
}

void GT_delq(void *model) {
  int error = GRBdelq((GRBmodel*) model);
  assert(!error);
}

int GT_solve(THDoubleTensor *rx, void *model) {
  GRBmodel *model_ = (GRBmodel*) model;
  int error = GRBoptimize(model);
  assert(!error);

  int status = GT_getintattr(model, "Status");
  int nVars = THDoubleTensor_size(rx, 0);

  int *idx = (int*) malloc(nVars * sizeof(int));
  for (int i = 0; i < nVars; i++) {
    idx[i] = i;
  }

  double *rx_ = THDoubleTensor_data(rx);
  error = GRBgetdblattrlist(model, "X", nVars, idx, rx_);
  assert(!error);

  return status;
}

void GT_free(void *env, void *model) {
  int error;
  if (model) {
    error = GRBfreemodel((GRBmodel*) model);
    assert(!error);
  }
  if (env) {
    GRBfreeenv((GRBenv*) env);
  }
}
