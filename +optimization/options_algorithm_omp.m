%
% superclass for all orthogonal matching pursuit (OMP) options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2019-09-17
%
classdef options_algorithm_omp < optimization.options_algorithm

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_algorithm_omp( rel_RMSE, N_iterations_max )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures valid rel_RMSE
            % superclass ensures valid N_iterations_max

            %--------------------------------------------------------------
            % 2.) create OMP options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@optimization.options_algorithm( rel_RMSE, N_iterations_max );

        end % function objects = options_algorithm_omp( rel_RMSE, N_iterations_max )

	end % methods

end % classdef options_algorithm_omp < optimization.options_algorithm
