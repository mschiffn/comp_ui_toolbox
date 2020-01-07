%
% superclass for all compressive sampling matching pursuit (CoSaMP) options
%
% author: Martin F. Schiffner
% date: 2019-09-22
% modified: 2020-01-03
%
classdef algorithm_cosamp < regularization.options.algorithm

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sparsity ( 1, 1 ) double { mustBePositive, mustBeInteger } = 10	% relative root-mean squared error

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = algorithm_cosamp( rel_RMSE, N_iterations_max, sparsity )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures valid rel_RMSE
            % superclass ensures valid N_iterations_max
            % property validation functions ensure valid sparsity

            % multiple rel_RMSE / single q
            if ~isscalar( rel_RMSE ) && isscalar( sparsity )
                sparsity = repmat( sparsity, size( rel_RMSE ) );
            end

            %--------------------------------------------------------------
            % 2.) create CoSaMP options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.algorithm( rel_RMSE, N_iterations_max );

            % iterate CoSaMP options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).sparsity = sparsity( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = algorithm_cosamp( rel_RMSE, N_iterations_max )

	end % methods

end % classdef algorithm_cosamp < regularization.options.algorithm
