#include <gurobi_c.h>

#include <TH/TH.h>

#undef NDEBUG
#include <assert.h>
#include <string.h>

void* /* GRBenv* */ loadenv(const char* logfilename, int outputFlag) {
  GRBenv *env = (GRBenv*) malloc(sizeof(GRBenv*));
  int error = GRBloadenv(&env, logfilename);
  assert(!error);

  int OutputFlag = 0;
  error = GRBsetintparam(env, "OutputFlag", outputFlag);
  assert(!error);

  return env;
}

void* /* GRBmodel* */ newmodel(void *env, const char *name, THDoubleTensor *obj,
                               THDoubleTensor *lb, THDoubleTensor *ub) {
  GRBmodel *model = (GRBmodel*) malloc(sizeof(GRBmodel*));
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

int getintattr(void *model, const char *name) {
  GRBmodel *model_ = (GRBmodel*) model;
  int attr;
  int error = GRBgetintattr(model, name, &attr);
  assert(!error);
  return attr;
}

void addconstr(void *model, int nnz, THIntTensor *cind, THDoubleTensor *cval,
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

int solve(THDoubleTensor *rx, void *model) {
  GRBmodel *model_ = (GRBmodel*) model;
  int error = GRBoptimize(model);
  assert(!error);

  int status = getintattr(model, "Status");
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
