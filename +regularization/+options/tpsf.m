%
% superclass for all TPSF options
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-01-03
%
classdef tpsf < regularization.options.common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        indices ( :, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 1024	% indices of TPSF

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = tpsf( options_common, indices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.common
            if ~isa( options_common, 'regularization.options.common' )
                errorStruct.message = 'options_common must be regularization.options.common!';
                errorStruct.identifier = 'tpsf:NoCommonOptions';
                error( errorStruct );
            end

            % ensure cell array for indices
            if ~iscell( indices )
                indices = { indices };
            end

            % multiple options_common / single indices
            if ~isscalar( options_common ) && isscalar( indices )
                indices = repmat( indices, size( options_common ) );
            end

            % single options_common / multiple indices
            if isscalar( options_common ) && ~isscalar( indices )
                options_common = repmat( options_common, size( indices ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( options_common, indices );

            %--------------------------------------------------------------
            % 2.) create TPSF options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.common( [ options_common.momentary ], [ options_common.tgc ], [ options_common.dictionary ], [ options_common.normalization ] );

            % reshape TPSF options
            objects = reshape( objects, size( options_common ) );

            % iterate TPSF options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).indices = indices{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = tpsf( options_common, indices )

	end % methods

end % classdef tpsf < regularization.options.common
