"""
Companion decorator for the PrsoxrLoader class. 

"""

def process_vars_properties(cls):
    """
    A class decorator that automatically creates properties for each key
    in a class-level dictionary called 'process_vars'.
    """
    # Get the dictionary from the class object itself.
    if hasattr(cls, 'process_vars') and isinstance(cls.process_vars, dict):
        process_vars_keys = cls.process_vars.keys()

        for key in process_vars_keys:
            # We still need to capture the 'key' variable for each loop iteration.
            # The default argument trick (k=key) is perfect for this.
            def getter(instance, k=key):
                """Reads the value from instance.process_vars[k]."""
                return instance.process_vars[k]

            def setter(instance, value, k=key):
                """Sets the value in instance.process_vars[k]."""
                #print(f"--- Setting '{k}' to '{value}'. Updating process_vars dictionary. ---")
                instance.process_vars[k] = value
                if 'reprocess_vars' not in k:
                    instance.process_vars['reprocess_vars'] = True

            # Use setattr() to attach the new property directly to the class.
            setattr(cls, key, property(getter, setter))

    return cls