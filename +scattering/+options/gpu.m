%
% abstract superclass for all GPU options
%
% author: Martin F. Schiffner
% date: 2019-07-09
% modified: 2020-01-18
%
classdef (Abstract) gpu < scattering.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = gpu( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create GPU options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.template( size );

        end % function objects = gpu( size )

	end % methods

end % classdef (Abstract) gpu < scattering.options.template
