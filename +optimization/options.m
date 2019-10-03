%
% superclass for all optimization options
%
% author: Martin F. Schiffner
% date: 2019-05-29
% modified: 2019-09-24
%
classdef options

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        normalization ( 1, 1 ) optimization.options.normalization { mustBeNonempty } = optimization.options.normalization_threshold( 1 ) % normalization options
        algorithm ( 1, 1 ) optimization.options.algorithm { mustBeNonempty } = optimization.options.algorithm_spgl1( 0.3, 1e3, 1 )	% algorithm options
        reweighting ( 1, 1 ) optimization.options.reweighting { mustBeNonempty } = optimization.options.reweighting_off             % reweighting options
        warm_start ( 1, 1 ) optimization.options.warm_start { mustBeNonempty } = optimization.options.warm_start_off                % warm start options

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no input argument
            if nargin == 0
                return;
            end

            % iterate arguments
            for index_arg = 2:numel( varargin )
                % multiple varargin{ 1 } / single varargin{ index_arg }
                if ~isscalar( varargin{ 1 } ) && isscalar( varargin{ index_arg } )
                    varargin{ index_arg } = repmat( varargin{ index_arg }, size( varargin{ 1 } ) );
                end
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create optimization options
            %--------------------------------------------------------------
            % repeat default optimization options
            objects = repmat( objects, size( varargin{ 1 } ) );

            % iterate optimization options
            for index_object = 1:numel( objects )

                % iterate arguments
                for index_arg = 1:numel( varargin )

                    if isa( varargin{ index_arg }, 'optimization.options.normalization' )

                        %--------------------------------------------------
                        % a) normalization options
                        %--------------------------------------------------
                        objects( index_object ).normalization = varargin{ index_arg }( index_object );

                    elseif isa( varargin{ index_arg }, 'optimization.options.algorithm' )

                        %--------------------------------------------------
                        % b) algorithm options
                        %--------------------------------------------------
                        objects( index_object ).algorithm = varargin{ index_arg }( index_object );

                    elseif isa( varargin{ index_arg }, 'optimization.options.reweighting' )

                        %--------------------------------------------------
                        % c) reweighting options
                        %--------------------------------------------------
                        objects( index_object ).reweighting = varargin{ index_arg }( index_object );

                    elseif isa( varargin{ index_arg }, 'optimization.options.warm_start' )

                        %--------------------------------------------------
                        % d) warm start options
                        %--------------------------------------------------
                        objects( index_object ).warm_start = varargin{ index_arg }( index_object );

                    else

                        %--------------------------------------------------
                        % e) unknown class
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                        errorStruct.identifier = 'options:UnknownClass';
                        error( errorStruct );

                    end % if isa( varargin{ index_arg }, 'optimization.options.normalization' )

                end % for index_arg = 1:numel( varargin )

            end % for index_object = 1:numel( objects )

        end % function objects = options( varargin )

	end % methods

end % classdef options
