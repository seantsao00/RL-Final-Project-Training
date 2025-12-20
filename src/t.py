import time


class Thermostat:
    """
    The class manages temperature control, including setting and retrieving the
    target temperature, adjusting the mode, and simulating temperature operation.
    """

    def __init__(self, current_temperature, target_temperature, mode):
        """
        initialize instances of the Thermostat class, including the current
        temperature, target temperature, and operating mode.
        :param current_temperature: float
        :param target_temperature: float
        :param mode: str, the work mode
        """
        self.current_temperature = current_temperature
        self.target_temperature = target_temperature
        self.mode = mode

    def get_target_temperature(self):
        """
        Get the target temperature of an instance of the Thermostat class.
        :return self.target_temperature: float
        >>> thermostat = Thermostat(20.4, 37.5, 'heat')
        >>> thermostat.get_target_temperature()
        37.5
        """
        return self.target_temperature

    def set_target_temperature(self, temperature):
        """
        Set the target temperature
        :param temperature: float, the target temperature
        >>> thermostat = Thermostat(20.4, 37.5, 'heat')
        >>> thermostat.set_target_temperature(37.6)
        >>> thermostat.target_temperature
        37.6
        """
        self.target_temperature = temperature

    def get_mode(self):
        """
        Get the current work mode
        :return mode: str, working mode. only ['heat', 'cool']
        """
        return self.mode

    def set_mode(self, mode):
        """
        Set the current work mode
        :param mode: str, working mode. only ['heat', 'cool']
        >>> thermostat = Thermostat(20.4, 37.5, 'heat')
        >>> thermostat.set_mode('cool')
        >>> thermostat.mode
        'cool'
        """
        if mode not in ['heat', 'cool']:
            raise ValueError("Mode must be either 'heat' or 'cool'")
        self.mode = mode

    def auto_set_mode(self):
        """
        Automatically set the operating mode by comparing with the current
        temperature and target temperature.
        >>> thermostat = Thermostat(20.4, 37.5, 'cool')
        >>> thermostat.auto_set_mode()
        >>> thermostat.mode
        'heat'
        """
        if self.current_temperature < self.target_temperature:
            self.mode = 'heat'
        else:
            self.mode = 'cool'

    def auto_check_conflict(self):
        """
        Check if there is a conflict between the operating mode and the
        relationship between the current temperature and the target temperature.
        If there is a conflict, the operating mode will be adjusted automatically.
        :return: True if no conflict, False otherwise.
        >>> thermostat = Thermostat(20.4, 37.5, 'cool')
        >>> thermostat.auto_check_conflict()
        False
        >>> thermostat.mode
        'heat'
        """
        if self.current_temperature < self.target_temperature and self.mode != 'heat':
            self.auto_set_mode()
            return False
        if self.current_temperature > self.target_temperature and self.mode != 'cool':
            self.auto_set_mode()
            return False
        return True

    def simulate_operation(self):
        """
        Simulate the operation of Thermostat.
        :return time: int, the time it took to complete the simulation.
        >>> thermostat = Thermostat(20.4, 37.5, 'cool')
        >>> thermostat.simulate_operation()
        18
        """
        self.auto_set_mode()
        elapsed_time = 0

        if self.mode == 'heat':
            while self.current_temperature < self.target_temperature:
                self.current_temperature += 1
                elapsed_time += 1
        else:  # mode == 'cool'
            while self.current_temperature > self.target_temperature:
                self.current_temperature -= 1
                elapsed_time += 1

        return elapsed_time
