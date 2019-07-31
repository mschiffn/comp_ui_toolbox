%
% superclass for all active GPU options
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2019-07-29
%
classdef options_gpu_active < scattering.options_gpu

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        index ( 1, 1 ) uint8 { mustBeNonnegative, mustBeNonempty } = 0;	% index of GPU device

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_gpu_active( indices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid indices

            %--------------------------------------------------------------
            % 2.) create active GPU options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options_gpu( size( indices ) );

            % iterate active GPU options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).index = indices( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = options_gpu_active( indices )

	end % methods

end % classdef options_gpu_active < scattering.options_gpu
