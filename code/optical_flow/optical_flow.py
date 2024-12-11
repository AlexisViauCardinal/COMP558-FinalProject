import abc

class OpticalFlow:

    @abc.abstractmethod
    def track_frame(self, previous_frame, next_frame, previous_points, previous_points_desc = None, next_points = None, next_points_desc = None):
        '''
            
        '''
        pass