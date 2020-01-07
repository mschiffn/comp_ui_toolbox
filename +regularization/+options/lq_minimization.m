%
% superclass for all lq-minimization options
%
% author: Martin F. Schiffner
% date: 2019-05-29
% modified: 2020-01-06
%
classdef lq_minimization < regularization.options.common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        algorithm ( 1, 1 ) regularization.options.algorithm { mustBeNonempty } = regularization.options.algorithm_spgl1( 0.3, 1e3, 1 )	% algorithm options
        reweighting ( 1, 1 ) regularization.options.reweighting { mustBeNonempty } = regularization.options.reweighting_off             % reweighting options
        warm_start ( 1, 1 ) regularization.options.warm_start { mustBeNonempty } = regularization.options.warm_start_off                % warm start options

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = lq_minimization( options_common, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty options_common
            if isempty( options_common )
                options_common = regularization.options.common;
            end

            % ensure class regularization.options.common
            if ~isa( options_common, 'regularization.options.common' )
                errorStruct.message = 'options_common must be regularization.options.common!';
                errorStruct.identifier = 'options_tpsf:NoCommonOptions';
                error( errorStruct );
            end

            % iterate arguments
            for index_arg = 1:numel( varargin )
                % multiple options_common / single varargin{ index_arg }
                if ~isscalar( options_common ) && isscalar( varargin{ index_arg } )
                    varargin{ index_arg } = repmat( varargin{ index_arg }, size( options_common ) );
                end
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( options_common, varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create lq-minimization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.common( [ options_common.momentary ], [ options_common.tgc ], [ options_common.dictionary ], [ options_common.normalization ] );

            % reshape lq-minimization options
            objects = reshape( objects, size( options_common ) );

            % iterate lq-minimization options
            for index_object = 1:numel( objects )

                % iterate arguments
                for index_arg = 1:numel( varargin )

                    if isa( varargin{ index_arg }, 'regularization.options.algorithm' )

                        %--------------------------------------------------
                        % a) algorithm options
                        %--------------------------------------------------
                        objects( index_object ).algorithm = varargin{ index_arg }( index_object );

                    elseif isa( varargin{ index_arg }, 'regularization.options.reweighting' )

                        %--------------------------------------------------
                        % b) reweighting options
                        %--------------------------------------------------
                        objects( index_object ).reweighting = varargin{ index_arg }( index_object );

                    elseif isa( varargin{ index_arg }, 'regularization.options.warm_start' )

                        %--------------------------------------------------
                        % c) warm start options
                        %--------------------------------------------------
                        objects( index_object ).warm_start = varargin{ index_arg }( index_object );

                    else

                        %--------------------------------------------------
                        % d) unknown class
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                        errorStruct.identifier = 'lq_minimization:UnknownClass';
                        error( errorStruct );

                    end % if isa( varargin{ index_arg }, 'regularization.options.algorithm' )

                end % for index_arg = 1:numel( varargin )

            end % for index_object = 1:numel( objects )

        end % function objects = lq_minimization( options_common, varargin )

	end % methods

end % classdef lq_minimization < regularization.options.common
