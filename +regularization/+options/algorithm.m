%
% abstract superclass for all algorithm options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-01-17
%
classdef (Abstract) algorithm < regularization.options.template

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        rel_RMSE ( 1, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( rel_RMSE, 1 ) } = 0.3	% relative root-mean squared error
        N_iterations_max ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 1e3	% maximum number of iterations

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = algorithm( rel_RMSE, N_iterations_max )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid rel_RMSE
            % property validation functions ensure valid N_iterations_max

            % multiple rel_RMSE / single N_iterations_max
            if ~isscalar( rel_RMSE ) && isscalar( N_iterations_max )
                N_iterations_max = repmat( N_iterations_max, size( rel_RMSE ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( rel_RMSE, N_iterations_max );

            %--------------------------------------------------------------
            % 2.) create algorithm options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.template( size( rel_RMSE ) );

            % iterate algorithm options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).rel_RMSE = rel_RMSE( index_object );
                objects( index_object ).N_iterations_max = N_iterations_max( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = algorithm( rel_RMSE, N_iterations_max )

	end % methods

end % classdef (Abstract) algorithm < regularization.options.template
