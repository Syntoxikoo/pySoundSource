from dataclasses import make_dataclass, asdict

# Define the SourceArg dataclass
SourceArg = make_dataclass(
    "SourceArg",
    [
        ("ID", str),
        ("x", float),
        ("y", float),
        ("directivity", str),
        ("gain", float),
        ("orientation", float),
        ("radius", float),
        ("fmin", int),
        ("fmax", int),
    ],
)


class SourcesManager:
    """
    TODO:
    Create a method to save each attributes of each instance in a csv or txt
    """

    def __init__(self):
        self.instances = {}
        self.attributes = {}

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

        self.attributes[_id] = SourceArg(
            ID=_id,
            x=kwargs["azim_v"],
            y=kwargs["elev_v"],
            directivity=kwargs["directivity"],
            gain=kwargs["src_resp"],
            orientation=kwargs["orientation_v"][0],
            radius=kwargs["radius"],
            fmin=kwargs.get("fmin", 20),
            fmax=kwargs.get("fmax", 20000),
        )
        return instance

    def get_instance(self, _id):
        """Get an instance by its ID"""
        return self.instances.get(_id)

    def get_instances(self, _ids):
        """Get multiple instances by their IDs

        Args:
            id_list (list): A list of instance IDs

        Returns:
            list: List of instances corresponding to the provided IDs
        """
        return [self.instances.get(id) for id in _ids if id in self.instances]

    def get_attributes(self, _id):
        """Get the attributes for an instance by its ID"""
        return self.attributes.get(_id)

    def list_instances(self):
        """List all instance IDs"""
        return list(self.instances.keys())

    def get_attributes_dict(self, _id):
        """Get the attributes as a dictionary"""
        source_args = self.get_attributes(_id)
        if source_args:
            return asdict(source_args)
        return None

    def remove_instance(self, _id):
        """Remove an instance from the registry"""
        if _id in self.instances:
            del self.instances[_id]
        if _id in self.attributes:
            del self.attributes[_id]
