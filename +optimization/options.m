%
% superclass for all optimization options
%
% author: Martin F. Schiffner
% date: 2019-05-29
% modified: 2019-05-29
%
classdef options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        rel_RMSE ( 1, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( rel_RMSE, 1 ) } = 0.2	% relative root-mean squared error
        N_iterations_max ( 1, 1 ) double { mustBePositive, mustBeInteger } = 1e3                    % maximum number of iterations

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options( rel_RMSE, N_iterations_max )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no input argument
            if nargin == 0
                return;
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( rel_RMSE, N_iterations_max );

            %--------------------------------------------------------------
            % 2.) create optimization options
            %--------------------------------------------------------------
            % repeat default optimization options
            objects = repmat( objects, size( rel_RMSE ) );

            % iterate optimization options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).rel_RMSE = rel_RMSE( index_object );
                objects( index_object ).N_iterations_max = N_iterations_max( index_object );

            end % for index_object = 1:numel( spatial )

        end % function objects = options( rel_RMSE, N_iterations_max )

	end % methods

end % classdef options
