%
% abstract superclass for dictionary options
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-01-17
%
classdef (Abstract) dictionary < regularization.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = dictionary( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create dictionary options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.template( size );

        end % function objects = dictionary( size )

	end % methods

end % classdef (Abstract) dictionary < regularization.options.template
