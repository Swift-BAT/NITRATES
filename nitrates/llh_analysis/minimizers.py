import numpy as np
import abc
from scipy import optimize
import six


@six.add_metaclass(abc.ABCMeta)
class NLLH_Minimizer(object):
    # , metaclass=abc.ABCMeta

    # should remember to put in support for several seeds

    def __init__(self, minimizer_type):
        self.param_info_dict = {}
        # important info about each param
        # fixed = boolean fixed or not
        # value = either the fixed value or first guess
        # nuis = boolean whether it's a nuissance param or not
        PARAM_INFO_KEYS = ["fixed", "val", "bounds", "nuis", "trans"]

        self._setup()

    #     @property
    #     def nebins(self):
    #         return self._nebins

    @abc.abstractmethod
    def minimize(self, x0s=None, ret_res=False):
        pass

    def _setup(self, setup=None):
        if setup is None:
            self.setup = {"algorithm": "L-BFGS-B", "tol": 0.0001}

    def set_fixed_params(self, param_names, values=None, fixed=True):
        for i, pname in enumerate(param_names):
            self.param_info_dict[pname]["fixed"] = fixed
            if values is not None:
                self.param_info_dict[pname]["val"] = values[i]

        self.fixed_params = [
            pname for pname in self.param_names if self.param_info_dict[pname]["fixed"]
        ]
        self.free_params = [
            pname
            for pname in self.param_names
            if not self.param_info_dict[pname]["fixed"]
        ]
        self.nfree_params = len(self.free_params)

    def set_bounds(self, param_names, bounds):
        for i, pname in enumerate(param_names):
            self.param_info_dict[pname]["bounds"] = bounds[i]

    def set_trans(self, param_names, trans_types):
        for i, pname in enumerate(param_names):
            self.param_info_dict[pname]["trans"] = trans_types[i]

    def set_param_info_dict(self):
        self.param_info_dict = self.model.param_dict

        self.fixed_params = [
            pname for pname in self.param_names if self.param_info_dict[pname]["fixed"]
        ]

        for pname in self.param_names:
            self.param_info_dict[pname]["trans"] = None
            # if ('E' in pname) or ('A' in pname):
            #     self.param_info_dict[pname]['trans'] = 'log'
            # else:
            #     self.param_info_dict[pname]['trans'] = None

    def set_llh(self, llh_obj):
        self.llh_obj = llh_obj

        self.model = llh_obj.model

        self.nparams = self.model.nparams
        self.nfree_params = self.model.nparams

        self.param_names = self.model.param_names

        self.set_param_info_dict()

    def norm_param(self, x, pname):
        new_x = 0.0
        bnd0 = self.param_info_dict[pname]["bounds"][0]
        bnd1 = self.param_info_dict[pname]["bounds"][1]
        rng = bnd1 - bnd0
        new_x = (x - bnd0) / rng
        return new_x

    def unnorm_param(self, x, pname):
        new_x = 0.0
        bnd0 = self.param_info_dict[pname]["bounds"][0]
        bnd1 = self.param_info_dict[pname]["bounds"][1]
        rng = bnd1 - bnd0
        new_x = x * rng + bnd0
        return new_x

    def lognorm_param(self, x, pname):
        new_x = 0.0
        bnd0 = np.log10(self.param_info_dict[pname]["bounds"][0])
        bnd1 = np.log10(self.param_info_dict[pname]["bounds"][1])
        rng = bnd1 - bnd0
        new_x = (np.log10(x) - bnd0) / rng
        return new_x

    def unlognorm_param(self, x, pname):
        new_x = 0.0
        bnd0 = np.log10(self.param_info_dict[pname]["bounds"][0])
        bnd1 = np.log10(self.param_info_dict[pname]["bounds"][1])
        rng = bnd1 - bnd0
        new_x = x * rng + bnd0
        new_x = 10.0 ** (new_x)
        return new_x

    def trans_param(self, x, pname):
        trans_type = self.param_info_dict[pname]["trans"]
        if trans_type is None:
            return x
        elif trans_type == "log":
            return np.log10(x)
        elif trans_type == "norm":
            return self.norm_param(x, pname)
        elif trans_type == "lognorm":
            return self.lognorm_param(x, pname)
        else:
            print(("Bad trans type:", trans_type))
            return x

    def untrans_param(self, x, pname):
        trans_type = self.param_info_dict[pname]["trans"]
        if trans_type is None:
            return x
        elif trans_type == "log":
            return 10.0**x
        elif trans_type == "norm":
            return self.unnorm_param(x, pname)
        elif trans_type == "lognorm":
            return self.unlognorm_param(x, pname)
        else:
            print(("Bad trans type:", trans_type))
            return x

    def trans_params(self, x):
        x_new = []
        i = 0
        # for i, pname in enumerate(self.param_names):
        for pname in self.param_names:
            if pname not in self.fixed_params:
                x_new.append(self.trans_param(x[i], pname))
                i += 1
        return x_new

    def untrans_params(self, x):
        x_new = []
        i = 0
        # for i, pname in enumerate(self.param_names):
        for pname in self.param_names:
            if pname not in self.fixed_params:
                x_new.append(self.untrans_param(x[i], pname))
                i += 1
        return x_new

    def get_default_x0(self):
        x0 = []
        for pname in self.param_names:
            if pname not in self.fixed_params:
                x0.append(self.param_info_dict[pname]["val"])
        return x0

    def wrapper(self, x):
        params = {}
        ii = 0
        for pname in self.param_names:
            if pname not in self.fixed_params:
                params[pname] = self.untrans_param(x[ii], pname)
                #                 if self.param_info_dict[pname]['trans'] is None:
                #                     params[pname] = x[ii]
                #                 elif self.param_info_dict[pname]['trans'] == 'log':
                #                     params[pname] = 10.**x[ii]
                #                 elif self.param_info_dict[pname]['trans'] == 'norm':
                #                     params[pname] = self.unnorm_param(x[ii], pname)
                #                 elif self.param_info_dict[pname]['trans'] == 'lognorm':
                #                     params[pname] = self.unlognorm_param(x[ii], pname)
                ii += 1
            else:
                params[pname] = self.param_info_dict[pname]["val"]

        return -1.0 * self.llh_obj.get_logprob(params)


class NLLH_DualAnnealingMin(NLLH_Minimizer):
    # should remember to put in support for several seeds

    def __init__(self, minimizer_type="Dual_Annealing"):
        # important info about each param
        # fixed = boolean fixed or not
        # value = either the fixed value or first guess
        # nuis = boolean whether it's a nuissance param or not
        PARAM_INFO_KEYS = ["fixed", "val", "bounds", "nuis", "trans"]

        super(NLLH_DualAnnealingMin, self).__init__(minimizer_type)

    def set_trans2norm(self):
        for pname in self.param_names:
            trans_type = self.param_info_dict[pname]["trans"]
            if trans_type is None:
                self.param_info_dict[pname]["trans"] = "norm"
            elif trans_type == "log":
                self.param_info_dict[pname]["trans"] = "lognorm"

    def minimize(
        self, x0s=None, maxiter=int(5e2), maxfun=1e4, seed=None, norm_params=True
    ):
        if norm_params:
            self.set_trans2norm()

        lowers = []
        uppers = []
        for pname in self.param_names:
            if pname not in self.fixed_params:
                lowers.append(
                    self.trans_param(self.param_info_dict[pname]["bounds"][0], pname)
                )
                uppers.append(
                    self.trans_param(self.param_info_dict[pname]["bounds"][1], pname)
                )

        bounds = optimize.Bounds(np.array(lowers), np.array(uppers))

        bounds = np.array([lowers, uppers]).T
        print(("bounds shape: ", bounds.shape))
        print("bounds: ")
        print(bounds)

        if x0s is not None:
            x0s = np.array(self.trans_params(x0s))

        res = optimize.dual_annealing(
            self.wrapper, bounds, maxiter=maxiter, maxfun=maxfun, x0=x0s, seed=seed
        )

        bf_vals = self.untrans_params(res.x)
        bf_nllh = res.fun

        return bf_vals, bf_nllh, res


class NLLH_ScipyMinimize(NLLH_Minimizer):
    # should remember to put in support for several seeds

    def __init__(self, minimizer_type):
        # important info about each param
        # fixed = boolean fixed or not
        # value = either the fixed value or first guess
        # nuis = boolean whether it's a nuissance param or not
        PARAM_INFO_KEYS = ["fixed", "val", "bounds", "nuis", "trans"]

        super(NLLH_ScipyMinimize, self).__init__(minimizer_type)

    def minimize(self, x0s=None, ret_res=False):
        lowers = []
        uppers = []
        for pname in self.param_names:
            if pname not in self.fixed_params:
                lowers.append(
                    self.trans_param(self.param_info_dict[pname]["bounds"][0], pname)
                )
                uppers.append(
                    self.trans_param(self.param_info_dict[pname]["bounds"][1], pname)
                )

        bounds = optimize.Bounds(np.array(lowers), np.array(uppers))

        if x0s is None:
            x0s = ["default"]

        bf_vals = []
        bf_nllhs = []
        ress = []

        for x0 in x0s:
            if x0 == "default":
                x0 = self.get_default_x0()

            x_0 = np.array(self.trans_params(x0))

            res = optimize.minimize(
                self.wrapper, x_0, method=self.setup["algorithm"], bounds=bounds
            )

            bf_vals.append(self.untrans_params(res.x))
            bf_nllhs.append(res.fun)
            ress.append(res)

        return bf_vals, bf_nllhs, ress


class NLLH_ScipyMinimize_Wjacob(NLLH_Minimizer):
    # should remember to put in support for several seeds

    def __init__(self, minimizer_type):
        # important info about each param
        # fixed = boolean fixed or not
        # value = either the fixed value or first guess
        # nuis = boolean whether it's a nuissance param or not
        PARAM_INFO_KEYS = ["fixed", "val", "bounds", "nuis", "trans"]

        super(NLLH_ScipyMinimize_Wjacob, self).__init__(minimizer_type)

    def set_trans2none(self):
        for pname in self.param_names:
            self.param_info_dict[pname]["trans"] = None

    def jacob_wrapper(self, x):
        params = {}
        ii = 0
        for pname in self.param_names:
            if pname not in self.fixed_params:
                params[pname] = self.untrans_param(x[ii], pname)
                ii += 1
            else:
                params[pname] = self.param_info_dict[pname]["val"]

        return np.array(self.llh_obj.get_logprob_jacob(params))

    def minimize(self, x0s=None, ret_res=False):
        self.set_trans2none()

        lowers = []
        uppers = []
        for pname in self.param_names:
            if pname not in self.fixed_params:
                lowers.append(
                    self.trans_param(self.param_info_dict[pname]["bounds"][0], pname)
                )
                uppers.append(
                    self.trans_param(self.param_info_dict[pname]["bounds"][1], pname)
                )

        bounds = optimize.Bounds(np.array(lowers), np.array(uppers))

        if x0s is None:
            x0s = ["default"]

        bf_vals = []
        bf_nllhs = []
        ress = []

        for x0 in x0s:
            if x0 == "default":
                x0 = self.get_default_x0()

            x_0 = np.array(self.trans_params(x0))

            res = optimize.minimize(
                self.wrapper,
                x_0,
                method=self.setup["algorithm"],
                bounds=bounds,
                jac=self.jacob_wrapper,
            )

            bf_vals.append(self.untrans_params(res.x))
            bf_nllhs.append(res.fun)
            ress.append(res)

        return bf_vals, bf_nllhs, ress


class NLLH_NloptMinimize(NLLH_Minimizer):
    # should remember to put in support for several seeds

    def __init__(self):
        import nlopt

        # important info about each param
        # fixed = boolean fixed or not
        # value = either the fixed value or first guess
        # nuis = boolean whether it's a nuissance param or not
        PARAM_INFO_KEYS = ["fixed", "val", "bounds", "nuis", "trans"]

        super(NLLH_NloptMinimize, self).__init__("NLOPT")

    def set_trans2none(self):
        for pname in self.param_names:
            self.param_info_dict[pname]["trans"] = None

    def jacob_wrapper(self, x):
        params = {}
        ii = 0
        for pname in self.param_names:
            if pname not in self.fixed_params:
                params[pname] = self.untrans_param(x[ii], pname)
                ii += 1
            else:
                params[pname] = self.param_info_dict[pname]["val"]

        return np.array(self.llh_obj.get_logprob_jacob(params))

    def nlopt_wrapper(self, x, grad):
        if grad.size > 0:
            grad = self.jacob_wrapper(x)

        return self.wrapper(x)

    def minimize(self, x0s=None, ret_res=False):
        #         self.set_trans2none()

        lowers = []
        uppers = []
        for pname in self.param_names:
            if pname not in self.fixed_params:
                lowers.append(
                    self.trans_param(self.param_info_dict[pname]["bounds"][0], pname)
                )
                uppers.append(
                    self.trans_param(self.param_info_dict[pname]["bounds"][1], pname)
                )

        bounds = optimize.Bounds(np.array(lowers), np.array(uppers))

        if x0s is None:
            x0s = ["default"]

        bf_vals = []
        bf_nllhs = []
        ress = []

        self.Npars2min = len(self.param_names) - len(self.fixed_params)
        #         self.opt = nlopt.opt(nlopt.GD_MLSL, self.Npars2min)
        opt_local = nlopt.opt(nlopt.LD_SLSQP, self.Npars2min)
        self.opt = nlopt.opt(nlopt.GN_DIRECT_L_NOSCAL, self.Npars2min)
        #         self.opt = nlopt.opt(nlopt.GN_CRS2_LM, self.Npars2min)
        #         self.opt = nlopt.opt(nlopt.LN_COBYLA, self.Npars2min)
        self.opt.set_local_optimizer(opt_local)

        self.opt.set_min_objective(self.nlopt_wrapper)
        self.opt.set_lower_bounds(lowers)
        self.opt.set_upper_bounds(uppers)
        self.opt.set_ftol_abs(1e-3)
        self.opt.set_xtol_abs(1e-6)
        self.opt.set_xtol_rel(1e-5)

        for x0 in x0s:
            if x0 == "default":
                x0 = self.get_default_x0()

            x_0 = np.array(self.trans_params(x0))
            x = self.opt.optimize(x_0)
            min_nllh = self.opt.last_optimum_value()
            res = self.opt.last_optimize_result()

            bf_vals.append(self.untrans_params(x))
            bf_nllhs.append(min_nllh)
            ress.append(res)

        return bf_vals, bf_nllhs, ress


def imxy_grid_miner(miner, imx0, imy0, imx1, imy1, dimxy=0.002):
    imxs = np.arange(imx0, imx1, dimxy)
    imys = np.arange(imy0, imy1, dimxy)
    grids = np.meshgrid(imxs, imys)
    imxs = grids[0].ravel()
    imys = grids[1].ravel()

    param_list = []
    nllhs = []

    # print len(imxs), " grid points to minimize at"

    for i in range(len(imxs)):
        miner.set_fixed_params(["Signal_imx", "Signal_imy"], values=[imxs[i], imys[i]])
        params, nllh, ress = miner.minimize()
        param_list.append(params[0])
        nllhs.append(nllh)

    return param_list, nllhs, imxs, imys


def imxy_grid_miner_wimjacob(miner, imx0, imy0, imx1, imy1, dimxy=0.002, ret_ims=False):
    imxs = np.arange(imx0, imx1, dimxy)
    imys = np.arange(imy0, imy1, dimxy)
    grids = np.meshgrid(imxs, imys)
    imxs = grids[0].ravel()
    imys = grids[1].ravel()

    param_list = []
    nllhs = []

    print((len(imxs), " grid points to minimize at"))
    x0s = [miner.get_default_x0()]

    for i in range(len(imxs)):
        miner.set_fixed_params(["Signal_imx", "Signal_imy"], values=[imxs[i], imys[i]])
        miner.set_fixed_params(["Signal_imx", "Signal_imy"], fixed=False)
        miner.set_bounds(
            ["Signal_imx", "Signal_imy"],
            [(imxs[i] - dimxy, imxs[i] + dimxy), (imys[i] - dimxy, imys[i] + dimxy)],
        )
        params, nllh, ress = miner.minimize()
        param_list.append(params[0])
        nllhs.append(nllh[0])

    if ret_ims:
        return param_list, nllhs, imxs, imys
    return param_list, nllhs
