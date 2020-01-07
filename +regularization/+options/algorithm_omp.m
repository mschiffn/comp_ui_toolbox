%
% superclass for all orthogonal matching pursuit (OMP) options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-01-03
%
classdef algorithm_omp < regularization.options.algorithm

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = algorithm_omp( rel_RMSE, N_iterations_max )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures valid rel_RMSE
            % superclass ensures valid N_iterations_max

            %--------------------------------------------------------------
            % 2.) create OMP options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.algorithm( rel_RMSE, N_iterations_max );

        end % function objects = algorithm_omp( rel_RMSE, N_iterations_max )

	end % methods

end % classdef algorithm_omp < regularization.options.algorithm
