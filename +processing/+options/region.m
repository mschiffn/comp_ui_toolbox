%
% superclass for all region options
%
% author: Martin F. Schiffner
% date: 2020-01-08
% modified: 2020-01-08
%
classdef region

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        ROI ( 1, 1 ) math.orthotope { mustBeNonempty } = math.orthotope     % ROI to be inspected
        boundary_dB ( 1, 1 ) double { mustBeNegative, mustBeNonempty } = -6	% boundary value in dB

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = region( ROIs, boundaries_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.orthotope
            if ~isa( ROIs, 'math.orthotope' )
                errorStruct.message = 'ROIs must be math.orthotope!';
                errorStruct.identifier = 'region:NoOrthotopes';
                error( errorStruct );
            end

            % property validation functions ensure valid boundaries_dB

            % multiple ROIs / single boundaries_dB
            if ~isscalar( ROIs ) && isscalar( boundaries_dB )
                boundaries_dB = repmat( boundaries_dB, size( ROIs ) );
            end

            % single ROIs / multiple boundaries_dB
            if isscalar( ROIs ) && ~isscalar( boundaries_dB )
                ROIs = repmat( ROIs, size( boundaries_dB ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( ROIs, boundaries_dB );

            %--------------------------------------------------------------
            % 2.) create region options
            %--------------------------------------------------------------
            % repeat default region options
            objects = repmat( objects, size( boundaries_dB ) );

            % iterate region options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).ROI = ROIs( index_object );
                objects( index_object ).boundary_dB = boundaries_dB( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = region( ROIs, boundaries_dB )

	end % methods

end % classdef region
