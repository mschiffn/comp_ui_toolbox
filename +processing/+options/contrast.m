%
% superclass for all contrast options
%
% author: Martin F. Schiffner
% date: 2020-01-30
% modified: 2020-01-30
%
classdef contrast

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        ROI_ref ( 1, 1 ) math.orthotope { mustBeNonempty } = math.orthotope         % reference region of interest
        ROI_noise ( 1, 1 ) math.orthotope { mustBeNonempty } = math.orthotope       % noisy region of interest
        dynamic_range_dB ( 1, 1 ) double { mustBePositive, mustBeNonempty } = 60	% limit for dynamic range (dB)

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = contrast( ROIs_ref, ROIs_noise )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure class math.orthotope for ROIs_ref and ROIs_noise

            % ensure equal subclasses of physical_values.length
%             auxiliary.mustBeEqualSubclasses( 'physical_values.length', ROIs_ref.intervals.lb );

            % multiple ROIs_ref / single ROIs_noise
            if ~isscalar( ROIs_ref ) && isscalar( ROIs_noise )
                ROIs_noise = repmat( ROIs_noise, size( ROIs_ref ) );
            end

            % single ROIs_ref / multiple ROIs_noise
            if isscalar( ROIs_ref ) && ~isscalar( ROIs_noise )
                ROIs_ref = repmat( ROIs_ref, size( ROIs_noise ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( ROIs_ref, ROIs_noise );

            %--------------------------------------------------------------
            % 2.) create contrast options
            %--------------------------------------------------------------
            % repeat default contrast options
            objects = repmat( objects, size( ROIs_ref ) );

            % iterate contrast options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).ROI_ref = ROIs_ref( index_object );
                objects( index_object ).ROI_noise = ROIs_noise( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = contrast( ROIs_ref, ROIs_noise )

	end % methods

end % classdef contrast
