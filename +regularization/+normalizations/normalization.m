%
% abstract superclass for all normalization options
%
% author: Martin F. Schiffner
% date: 2019-08-10
% modified: 2020-02-17
%
classdef (Abstract) normalization < regularization.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = normalization( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create normalization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.template( size );

        end % function objects = normalization( size )

        %------------------------------------------------------------------
        % apply normalizations
        %------------------------------------------------------------------
        function weightings = apply( normalizations, weightings )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.normalizations.normalization
            if ~isa( normalizations, 'regularization.normalizations.normalization' )
                errorStruct.message = 'normalizations must be regularization.normalizations.normalization!';
                errorStruct.identifier = 'get_LTs:NoNormalizations';
                error( errorStruct );
            end

            % ensure class linear_transforms.weighting
            if ~isa( weightings, 'linear_transforms.weighting' )
                errorStruct.message = 'weightings must be linear_transforms.weighting!';
                errorStruct.identifier = 'get_LTs:NoWeightings';
                error( errorStruct );
            end

            % multiple normalizations / single weightings
            if ~isscalar( normalizations ) && isscalar( weightings )
                weightings = repmat( weightings, size( normalizations ) );
            end

            % single normalizations / multiple weightings
            if isscalar( normalizations ) && ~isscalar( weightings )
                normalizations = repmat( normalizations, size( weightings ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( normalizations, weightings );

            %--------------------------------------------------------------
            % 2.) apply normalizations
            %--------------------------------------------------------------
            % iterate normalizations
            for index_object = 1:numel( normalizations )

                % create linear transform (scalar)
                weightings( index_object ) = apply_scalar( normalizations( index_object ), weightings( index_object ) );

            end % for index_object = 1:numel( normalizations )

        end % function weightings = apply( normalizations, weightings )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % apply normalization (scalar)
        %------------------------------------------------------------------
        weighting = apply_scalar( normalization, weighting )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) normalization < regularization.options.template
