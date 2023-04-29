import numpy as np


class Norm_3D(object):
    def __init__(self, mean, cov):
        self.Ndim = len(mean)
        self.set_mean(mean)
        self.set_cov(cov)

    def set_mean(self, mean):
        self.mean = mean

    def set_cov(self, cov):
        self.cov = cov
        self.sigx2 = self.cov[0, 0]
        self.sigy2 = self.cov[1, 1]
        self.sigz2 = self.cov[2, 2]
        self.sigx = np.sqrt(self.cov[0, 0])
        self.sigy = np.sqrt(self.cov[1, 1])
        self.sigz = np.sqrt(self.cov[2, 2])
        self.rhoxy = self.cov[0, 1] / (self.sigx * self.sigy)
        self.rhoxz = self.cov[0, 2] / (self.sigx * self.sigz)
        self.rhoyz = self.cov[1, 2] / (self.sigy * self.sigz)

        self.rho_hyp = self.rhoxy**2 + self.rhoxz**2 + self.rhoyz**2
        self.rho_prod = self.rhoxy * self.rhoxz * self.rhoyz
        self.cov_det = -(self.sigx2 * self.sigy2 * self.sigz2) * (
            self.rho_hyp - 2 * self.rho_prod - 1.0
        )
        self.inv_cov_coef = 1.0 / (self.rho_hyp - 2 * self.rho_prod - 1.0)

        self._logpdf_x0 = np.log(((2.0 * np.pi) ** 3) * self.cov_det)

        self.set_hess_log_pdf()

    def logpdf(self, x_, y_, z_):
        x = x_ - self.mean[0]
        y = y_ - self.mean[1]
        z = z_ - self.mean[2]

        exp_term = (
            ((self.rhoyz**2) - 1.0) * (x**2 / self.sigx2)
            + ((self.rhoxz**2) - 1.0) * (y**2 / self.sigy2)
            + ((self.rhoxy**2) - 1.0) * (z**2 / self.sigz2)
            + 2
            * (self.rhoxy - self.rhoxz * self.rhoyz)
            * (x * y / (self.sigx * self.sigy))
            + 2
            * (self.rhoxz - self.rhoxy * self.rhoyz)
            * (x * z / (self.sigx * self.sigz))
            + 2
            * (self.rhoyz - self.rhoxy * self.rhoxz)
            * (y * z / (self.sigy * self.sigz))
        )

        self._log_pdf = -0.5 * (self._logpdf_x0 + self.inv_cov_coef * exp_term)

        return self._log_pdf

    def jacob_log_pdf(self, x_, y_, z_):
        x = x_ - self.mean[0]
        y = y_ - self.mean[1]
        z = z_ - self.mean[2]

        jacob = np.zeros(3)

        jacob[0] = (
            (self.rhoyz**2 - 1.0) * (x / self.sigx2)
            + (self.rhoxy - self.rhoxz * self.rhoyz) * (y / (self.sigx * self.sigy))
            + (self.rhoxz - self.rhoxy * self.rhoyz) * (z / (self.sigx * self.sigz))
        )
        jacob[1] = (
            (self.rhoxz**2 - 1.0) * (y / self.sigy2)
            + (self.rhoxy - self.rhoxz * self.rhoyz) * (x / (self.sigx * self.sigy))
            + (self.rhoyz - self.rhoxy * self.rhoxz) * (z / (self.sigy * self.sigz))
        )
        jacob[2] = (
            (self.rhoxy**2 - 1.0) * (z / self.sigz2)
            + (self.rhoxz - self.rhoxy * self.rhoyz) * (x / (self.sigx * self.sigz))
            + (self.rhoyz - self.rhoxy * self.rhoxz) * (y / (self.sigy * self.sigz))
        )

        jacob *= -self.inv_cov_coef

        return jacob

    def set_hess_log_pdf(self):
        hess = np.zeros((3, 3))

        hess[0, 0] = (self.rhoyz**2 - 1.0) / (self.sigx2)
        hess[1, 1] = (self.rhoxz**2 - 1.0) / (self.sigy2)
        hess[2, 2] = (self.rhoxy**2 - 1.0) / (self.sigz2)

        hess[0, 1] = (self.rhoxy - self.rhoxz * self.rhoyz) / (self.sigx * self.sigy)
        hess[1, 0] += hess[0, 1]
        hess[1, 2] = (self.rhoyz - self.rhoxz * self.rhoxy) / (self.sigz * self.sigy)
        hess[2, 1] += hess[1, 2]
        hess[0, 2] = (self.rhoxz - self.rhoxy * self.rhoyz) / (self.sigx * self.sigz)
        hess[2, 0] += hess[0, 2]

        hess *= -self.inv_cov_coef
        self.hess_log_pdf = hess

        return self.hess_log_pdf


class Norm_2D(object):
    def __init__(self, mean, cov):
        self.Ndim = len(mean)
        self.set_mean(mean)
        self.set_cov(cov)

    def set_mean(self, mean):
        self.mean = mean

    def set_cov(self, cov):
        self.cov = cov
        self.sigx2 = self.cov[0, 0]
        self.sigy2 = self.cov[1, 1]
        self.sigx = np.sqrt(self.cov[0, 0])
        self.sigy = np.sqrt(self.cov[1, 1])
        self.rho = self.cov[0, 1] / (self.sigx * self.sigy)

        self.pdf_coef = 1.0 / (
            2.0 * np.pi * self.sigx * self.sigy * np.sqrt(1.0 - self.rho**2)
        )
        self.pdf_coef_log = np.log(self.pdf_coef)
        self.exp_term_coef = -1.0 / (2.0 * (1.0 - self.rho**2))

        #         self.rho_hyp = self.rhoxy**2+self.rhoxz**2+self.rhoyz**2
        #         self.rho_prod = self.rhoxy*self.rhoxz*self.rhoyz
        #         self.cov_det = -(self.sigx2*self.sigy2*self.sigz2)*(self.rho_hyp - 2*self.rho_prod - 1.0)
        #         self.inv_cov_coef = 1./(self.rho_hyp - 2*self.rho_prod - 1.0)

        #         self._logpdf_x0 = np.log(((2.*np.pi)**3)*self.cov_det)

        self.set_hess_log_pdf()

    def logpdf(self, x_, y_):
        x = x_ - self.mean[0]
        y = y_ - self.mean[1]

        exp_term = (
            ((x**2) / self.sigx2)
            + ((y**2) / self.sigy2)
            - 2.0 * self.rho * x * y / (self.sigx * self.sigy)
        )
        exp_term *= self.exp_term_coef

        self._log_pdf = self.pdf_coef_log + exp_term

        return self._log_pdf

    def jacob_log_pdf(self, x_, y_):
        x = x_ - self.mean[0]
        y = y_ - self.mean[1]

        jacob = np.zeros(2)

        jacob[0] = 2 * x / self.sigx2 - 2.0 * self.rho * y / (self.sigx * self.sigy)
        jacob[1] = 2 * y / self.sigy2 - 2.0 * self.rho * x / (self.sigx * self.sigy)
        jacob *= self.exp_term_coef

        return jacob

    def set_hess_log_pdf(self):
        hess = np.zeros((2, 2))

        hess[0, 0] = self.exp_term_coef * 2.0 / (self.sigx2)
        hess[1, 1] = self.exp_term_coef * 2.0 / (self.sigy2)

        hess[0, 1] = -2.0 * self.exp_term_coef * self.rho / (self.sigx * self.sigy)
        hess[1, 0] += hess[0, 1]

        self.hess_log_pdf = hess

        return self.hess_log_pdf


class Norm_1D(object):
    def __init__(self, mean, cov):
        self.Ndim = 1
        self.set_mean(mean)
        self.set_cov(cov)

    def set_mean(self, mean):
        self.mean = mean

    def set_cov(self, cov):
        self.sig2 = cov
        self.sig = np.sqrt(cov)

        self.pdf_log_term = -np.log(self.sig) - np.log(2.0 * np.pi) / 2.0
        self.set_hess_log_pdf()

    def logpdf(self, x_):
        x = x_ - self.mean

        exp_term = -(x**2) / (2.0 * self.sig2)

        self._log_pdf = self.pdf_log_term + exp_term

        return self._log_pdf

    def jacob_log_pdf(self, x_):
        x = x_ - self.mean

        jacob = np.array([-x / (self.sig2)])

        return jacob

    def set_hess_log_pdf(self):
        hess = -1.0 / self.sig2

        self.hess_log_pdf = np.array([[hess]])

        return self.hess_log_pdf
