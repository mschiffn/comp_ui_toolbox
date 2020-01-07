%
% superclass for all inactive time gain compensation (TGC) options
%
% author: Martin F. Schiffner
% date: 2019-12-15
% modified: 2020-01-03
%
classdef tgc_off < regularization.options.tgc

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = tgc_off( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers

            %--------------------------------------------------------------
            % 2.) create inactive TGC options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.tgc( size );

        end % function objects = tgc_off( varargin )

	end % methods

end % classdef tgc_off < regularization.options.tgc
