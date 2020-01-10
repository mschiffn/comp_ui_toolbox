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
        boundary_dB ( 1, 1 ) double { mustBeNegative, mustBeNonempty } = -6	% boundary value in dB
        ROI ( 1, 1 ) math.orthotope { mustBeNonempty } = math.orthotope %

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = region( boundaries_dB, ROIs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid boundaries_dB

            % ensure class math.orthotope
            if ~isa( ROIs, 'math.orthotope' )
                errorStruct.message = 'ROIs must be math.orthotope!';
                errorStruct.identifier = 'region:NoOrthotopes';
                error( errorStruct );
            end

            % multiple boundaries_dB / single ROIs
            if ~isscalar( boundaries_dB ) && isscalar( ROIs )
                ROIs = repmat( ROIs, size( boundaries_dB ) );
            end

            % single boundaries_dB / multiple ROIs
            if isscalar( boundaries_dB ) && ~isscalar( ROIs )
                boundaries_dB = repmat( boundaries_dB, size( ROIs ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( boundaries_dB, ROIs );

            %--------------------------------------------------------------
            % 2.) create region options
            %--------------------------------------------------------------
            % repeat default region options
            objects = repmat( objects, size( boundaries_dB ) );

            % iterate region options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).boundary_dB = boundaries_dB( index_object );
                objects( index_object ).ROI = ROIs( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = region( boundaries_dB, ROIs )

	end % methods

end % classdef region
