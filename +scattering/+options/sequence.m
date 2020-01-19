%
% abstract superclass for all sequence options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2020-01-18
%
classdef (Abstract) sequence < scattering.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create sequence options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.template( size );

        end % function objects = sequence( size )

	end % methods

end % classdef (Abstract) sequence < scattering.options.template
