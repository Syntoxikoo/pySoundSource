import pandas as pd


class SourcesManager:
    """
    Manager class for audio sources with methods to create, retrieve and manage source instances
    """

    def __init__(self):
        self.instances = {}
        # Initialize attributes as an empty DataFrame
        self.attributes = pd.DataFrame(
            columns=[
                "ID",
                "x",
                "y",
                "directivity",
                "gain",
                "orientation",
                "radius",
                "fmin",
                "fmax",
                "delay",
            ]
        )
        self.FieldSettings = {}

    def create_instance(self, _id, class_name, *args, **kwargs):
        """
        Create and store an instance with a unique identifier and its attributes.

        Args:
            _id: Unique identifier for this instance
            class_name: The class to instantiate
            *args: Arguments to pass to the class constructor
            source_args: SourceArg instance with the source attributes
            **kwargs: Additional arguments to pass to the class constructor

        Returns:
            The created instance
        """

        instance = class_name(*args, **kwargs)

        self.instances[_id] = instance

        if not self.instances[_id].directed:
            self.instances[_id].set_directivity(kwargs["directivity"])

        self.attributes.loc[_id] = [
            _id,
            kwargs["position_v"][0],
            kwargs["position_v"][1],
            kwargs["directivity"],
            kwargs["src_resp"],
            kwargs["orientation_v"][0],
            kwargs["radius"],
            kwargs.get("fmin", 20),
            kwargs.get("fmax", 20000),
            kwargs["delay"],
        ]
        self.Stored = self.attributes.copy()
        self.FieldSettings = {
            "grid_length": len(kwargs["azim_v"]),
            "nfft": kwargs["n_fft"],
            "fs": kwargs["fs"],
        }

        return instance

    def get_instance(self, _id):
        """Get an instance by its ID"""
        return self.instances.get(_id)

    def get_instances(self, _ids):
        """
        Get multiple instances by their IDs
        Args:
            id_list (list): A list of instance IDs

        Returns:
            list: List of instances corresponding to the provided IDs
        """
        return [self.instances.get(id) for id in _ids if id in self.instances]

    def get_attributes(self, _id):
        """Get the attributes for an instance by its ID"""
        if _id in self.attributes.index:
            return self.attributes.loc[_id]
        return None

    def get_attribute(self, _id, key, default=None):
        """Get a specific attribute for an instance by its ID and attribute key"""
        if _id in self.attributes.index and key in self.attributes.columns:
            return self.attributes.loc[_id, key]
        return default

    def list_instances(self):
        """List all instance IDs"""
        return list(self.instances.keys())

    def get_attributes_dict(self, _id):
        """Get the attributes as a dictionary"""
        if _id in self.attributes.index:
            return self.attributes.loc[_id].to_dict()
        return None

    def remove_instance(self, _id):
        """Remove an instance from the registry"""
        if _id in self.instances:
            del self.instances[_id]
        elif _id.lower() == "all":
            del self.instances
            self.instances = {}
            self.attributes = pd.DataFrame(columns=self.attributes.columns)
            self.Stored = self.attributes.copy()
        if _id in self.attributes.index:
            self.attributes = self.attributes.drop(_id)

    def save_attributes(self, filepath):
        """Save all attributes to a CSV file"""
        # Save with index to preserve IDs
        self.attributes.to_csv(filepath)

    def load_attributes(self, filepath):
        """Load attributes from a CSV file"""
        self.attributes = pd.read_csv(filepath, index_col=0)
