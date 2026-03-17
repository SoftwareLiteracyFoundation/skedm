from .aux_func import IsIterable


# EDM method
def EDM_params(self):
    """Validate parameters and data passed to .fit()"""

    # If noTime is True create time index in first column
    if self._noTime:
        self._time = array([i for i in range(1,self._Data.shape[1]+1)], dtype=int)
        self._Data.insert(0, "time", self._time)
    else:
        self._time = self._Data.iloc[:,0] # First column is time

    if not self.embedded:
        if self.tau == 0:
            raise RuntimeError(f"Validate() {self._name}:" + " tau must be non-zero.")
        if self._E < 1:
            raise RuntimeError(
                f"EDM_params() {self._name}:"
                + f" E = {self._E} is invalid "
                + "if embedded = False"
            )

    # if lib or pred are not provided default to all observations
    if self._lib is None:
        self._lib = [1, self._Data.shape[0]]

    if self._pred is None:
        self._pred = [1, self._Data.shape[0]]

    if not IsIterable(self._lib):
        raise RuntimeError(f"EDM_params() {self._name}: lib is not iterable.")

    if not IsIterable(self._pred):
        raise RuntimeError(f"EDM_params() {self._name}: pred is not iterable.")

    # Set knn default based on E and lib size, E embedded on num columns
    if self._name in ["Simplex", "CCM", "Multiview"]:
        # embedded = true: Set E to number of columns
        if self.embedded:
            self._E = len(self._columns)

        # knn not specified : knn set to E+1
        if self._knn < 1:
            self._knn = self._E + 1

            # if self.verbose:
            #    msg = f"{self._name} Validate(): Set knn = {self.knn}"
            #    print(msg, flush=True)

    if self._name == "SMap":
        # embedded = true: Set E to number of columns
        if self.embedded and len(self._columns):
            self._E = len(self._columns)

        if not self.embedded and len(self._columns) > 1:
            msg = (
                f"{self._name} EDM_params(): Multivariable S-Map "
                + "must use embedded = True to ensure data/dimension "
                + "correspondance."
            )
            raise RuntimeError(msg)
