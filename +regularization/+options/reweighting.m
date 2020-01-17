%
% abstract superclass for all reweighting options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-01-17
%
classdef (Abstract) reweighting < regularization.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = reweighting( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create reweighting options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.template( size );

        end % function objects = reweighting( size )

	end % methods

end % classdef (Abstract) reweighting < regularization.options.template
