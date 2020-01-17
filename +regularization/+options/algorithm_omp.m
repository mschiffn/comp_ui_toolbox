%
% superclass for all orthogonal matching pursuit (OMP) options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-01-16
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

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( algorithms_omp )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.algorithm_omp
            if ~isa( algorithms_omp, 'regularization.options.algorithm_omp' )
                errorStruct.message = 'algorithms_omp must be regularization.options.algorithm_omp!';
                errorStruct.identifier = 'string:NoOptionsOMP';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "OMP (q = 0)"
            strs_out = repmat( "OMP (q = 0)", size( algorithms_omp ) );

        end % function strs_out = string( algorithms_omp )

	end % methods

end % classdef algorithm_omp < regularization.options.algorithm
