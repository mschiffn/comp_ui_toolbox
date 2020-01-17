%
% abstract superclass for all normalization options
%
% author: Martin F. Schiffner
% date: 2019-08-10
% modified: 2020-01-17
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

	end % methods

end % classdef (Abstract) normalization < regularization.options.template
