%
% superclass for all optimization options
%
% author: Martin F. Schiffner
% date: 2019-05-29
% modified: 2019-08-10
%
classdef options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        rel_RMSE ( 1, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( rel_RMSE, 1 ) } = 0.3	% relative root-mean squared error
        N_iterations_max ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 1e3	% maximum number of iterations
        normalization ( 1, 1 ) optimization.options_normalization { mustBeNonempty } = optimization.options_normalization_off % normalization options

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options( rel_RMSE, N_iterations_max, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no input argument
            if nargin == 0
                return;
            end

            % property validation functions ensure valid rel_RMSE
            % property validation functions ensure valid N_iterations_max

            % multiple rel_RMSE / single N_iterations_max
            if ~isscalar( rel_RMSE ) && isscalar( N_iterations_max )
                N_iterations_max = repmat( N_iterations_max, size( rel_RMSE ) );
            end

            % iterate arguments
            for index_arg = 1:numel( varargin )
                % multiple rel_RMSE / single varargin{ index_arg }
                if ~isscalar( rel_RMSE ) && isscalar( varargin{ index_arg } )
                    varargin{ index_arg } = repmat( varargin{ index_arg }, size( rel_RMSE ) );
                end
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( rel_RMSE, N_iterations_max, varargin{ : } );

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

                % iterate arguments
                for index_arg = 1:numel( varargin )

                    if isa( varargin{ index_arg }, 'optimization.options_normalization' )

                        %--------------------------------------------------
                        % a) normalization options
                        %--------------------------------------------------
                        objects( index_object ).normalization = varargin{ index_arg }( index_object );

                    else

                        %--------------------------------------------------
                        % b) unknown class
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                        errorStruct.identifier = 'options:UnknownClass';
                        error( errorStruct );

                    end % if isa( varargin{ index_arg }, 'optimization.options_normalization' )

                end % for index_arg = 1:numel( varargin )

            end % for index_object = 1:numel( objects )

        end % function objects = options( rel_RMSE, N_iterations_max, varargin )

	end % methods

end % classdef options
