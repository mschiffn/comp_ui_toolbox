%
% superclass for all projected profile options
%
% author: Martin F. Schiffner
% date: 2020-01-08
% modified: 2020-02-02
%
classdef profile

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        ROI ( 1, 1 ) math.orthotope { mustBeNonempty } = math.orthotope                                 % region of interest
        dim ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 1                       % profile dimension (projection along remaining dimensions)
        N_zeros_add ( 1, 1 ) double { mustBeNonnegative, mustBeInteger, mustBeNonempty } = 50           % number of padded zeros
        factor_interp ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 20            % interpolation factor
        setting_window ( 1, 1 ) auxiliary.setting_window { mustBeNonempty } = auxiliary.setting_window	% window settings

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = profile( ROIs, dims )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.orthotope
            if ~isa( ROIs, 'math.orthotope' )
                errorStruct.message = 'ROIs must be math.orthotope!';
                errorStruct.identifier = 'profile:NoOrthotopes';
                error( errorStruct );
            end

            % multiple ROIs / single dims
            if ~isscalar( ROIs ) && isscalar( dims )
                dims = repmat( dims, size( ROIs ) );
            end

            % single ROIs / multiple dims
            if isscalar( ROIs ) && ~isscalar( dims )
                ROIs = repmat( ROIs, size( dims ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( ROIs, dims );

            %--------------------------------------------------------------
            % 2.) create profile options
            %--------------------------------------------------------------
            % repeat default profile options
            objects = repmat( objects, size( ROIs ) );

            % iterate profile options
            for index_object = 1:numel( objects )

                % ensure equal subclasses of physical_values.length
                auxiliary.mustBeEqualSubclasses( 'physical_values.length', ROIs( index_object ).intervals.lb );

                % ensure valid profile dimension
                if ~ismember( dims( index_object ), ( 1:ROIs( index_object ).N_dimensions ) )
                    errorStruct.message = sprintf( 'dims( %d ) must be greater than or equal to 1 but smaller than or equalt to %d!', index_object, ROIs( index_object ).N_dimensions );
                    errorStruct.identifier = 'profile:InvalidDims';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).ROI = ROIs( index_object );
                objects( index_object ).dim = dims( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = profile( ROIs, dims )

	end % methods

end % classdef profile
