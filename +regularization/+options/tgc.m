%
% abstract superclass for all time gain compensation (TGC) options
%
% author: Martin F. Schiffner
% date: 2019-12-15
% modified: 2020-01-17
%
classdef (Abstract) tgc < regularization.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = tgc( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create TGC options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.template( size );

        end % function objects = tgc( size )

	end % methods

end % classdef (Abstract) tgc < regularization.options.template
